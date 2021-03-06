import tensorflow as tf
from utils import Droppath


class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, n_patches):
        super(SpatialGatingUnit, self).__init__()
        self.n_patches = n_patches

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Permute((2, 1)),
            tf.keras.layers.Dense(n_patches,
                                  kernel_initializer=tf.keras.initializers.truncated_normal(stddev=.001),
                                  bias_initializer=tf.keras.initializers.ones()
                                  ),
            tf.keras.layers.Permute((2, 1))
        ])

    @tf.function
    def call(self, inputs, **kwargs):
        u, v = tf.split(inputs, num_or_size_splits=2, axis=-1)
        v = self.forward(v)
        return u * v


class GmlpBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn, n_patches, survival_prob):
        super(GmlpBlock, self).__init__()
        self.d_mode = d_model
        self.d_ffn = d_ffn
        self.n_patches = n_patches
        self.survival_prob = survival_prob

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(d_ffn,
                                  kernel_initializer='lecun_normal'
                                  ),
            tf.keras.layers.Activation('gelu'),
            SpatialGatingUnit(self.n_patches),
            tf.keras.layers.Dense(d_model)
        ])
        self.droppath = Droppath(survival_prob) if survival_prob != 1. else tf.keras.layers.Layer()

    @tf.function
    def call(self, inputs, **kwargs):
        return self.droppath(self.forward(inputs)) + inputs


class Gmlp(tf.keras.models.Model):
    '''
    For image tasks, hidden layer have no dropout layer(.1 for NLP tasks)
    Input resolution : 256
    '''
    def __init__(self,
                 d_model,
                 d_ffn,
                 survival_prob,
                 num_classes,
                 n_layers=30,
                 input_res=224,
                 patch_res=16):
        super(Gmlp, self).__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.survival_prob = survival_prob
        self.num_classes = num_classes
        self.n_layers = n_layers
        if (input_res % patch_res) != 0:
            raise ValueError('size error')
        else:
            self.input_res = input_res
            self.patch_res = patch_res
            self.n_patches = int((tf.square(224) / tf.square(self.patch_res)).numpy())

        self.patch_projector = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.d_model,
                                   kernel_size=(self.patch_res, self.patch_res),
                                   strides=(self.patch_res, self.patch_res),
                                   padding='valid',
                                   activation='linear'
                                   ),
            tf.keras.layers.Reshape((self.n_patches, self.d_model))
        ])
        self.blocks = tf.keras.Sequential([
            GmlpBlock(self.d_model,
                      self.d_ffn,
                      self.n_patches,
                      self.survival_prob
                      ) for _ in range(self.n_layers)
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(self.num_classes,
                                  activation='softmax',
                                  kernel_initializer=tf.keras.initializers.zeros()
                                  )
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        patches = self.patch_projector(inputs)
        featuremap = self.blocks(patches)
        y = self.classifier(featuremap)
        return y