import tensorflow as tf


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

    @tf.function
    def call(self, inputs, **kwargs):
        stochastic_depth = tf.keras.backend.random_bernoulli(shape=(1,),
                                                             p=self.survival_prob
                                                             )
        return self.forward(inputs) * stochastic_depth + inputs


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
                 aug_res=224,
                 patch_res=16):
        super(Gmlp, self).__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.survival_prob = survival_prob
        self.num_classes = num_classes
        self.n_layers = n_layers
        if (aug_res % patch_res) != 0:
            raise ValueError('size error')
        else:
            self.aug_res = aug_res
            self.patch_res = patch_res
            self.n_patches = int((tf.square(224) / tf.square(self.patch_res)).numpy())

        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.015),
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224)
        ])
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
        aug = self.augmentation(inputs)
        patches = self.patch_projector(aug)
        featuremap = self.blocks(patches)
        y = self.classifier(featuremap)
        return y