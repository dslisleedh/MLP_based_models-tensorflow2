import tensorflow as tf
from utils import Droppath


class MlpBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_nodes :int,
                 recon_nodes :int,
                 dropout_rate :float):
        super(MlpBlock, self).__init__()
        self.n_nodes = n_nodes
        self.recon_nodes = recon_nodes
        self.dropout_rate = dropout_rate

        self.w1 = tf.keras.layers.Dense(self.n_nodes,
                                        activation='linear',
                                        kernel_initializer='lecun_normal'
                                        )
        self.w2 = tf.keras.layers.Dense(self.recon_nodes,
                                        activation='linear',
                                        kernel_initializer='lecun_normal'
                                        )

    def call(self, inputs, *args, **kwargs):
        return self.w2(tf.nn.gelu(self.w1(inputs)))


class MixerLayer(tf.keras.layers.Layer):
    def __init__(self,
                 dim_c:int,
                 dim_s:int,
                 projection_dim:int,
                 n_patches:int,
                 dropout_rate:float,
                 survival_prob
                 ):
        super(MixerLayer, self).__init__()
        self.dim_c = dim_c
        self.dim_s = dim_s
        self.projection_dim = projection_dim
        self.n_patches = n_patches
        self.survival_prob = survival_prob
        self.dropout_rate = dropout_rate

        self.D_c = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Permute((2, 1)),
            MlpBlock(self.dim_c, self.n_patches, self.dropout_rate),
            tf.keras.layers.Permute((2, 1))
        ])
        self.D_s = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            MlpBlock(self.dim_s, self.projection_dim, self.dropout_rate),
        ])
        self.droppath = Droppath(self.survival_prob) if self.survival_prob != 1. else tf.keras.layers.Layer()

    def call(self, inputs, *args, **kwargs):
        u = self.droppath(self.D_c(inputs)) + inputs
        y = self.droppath(self.D_s(u)) + u
        return y


class MlpMixer(tf.keras.models.Model):
    '''
    IMG input resolution : 256
    '''
    def __init__(self,
                 num_mixer_layers: int,
                 patch_res: int,
                 hidden_size_c: int,
                 dim_dc: int,
                 dim_ds: int,
                 num_labels: int,
                 dropout_rate: float = 0.0,
                 stochastic_depth: float = 0.1,
                 ):
        super(MlpMixer, self).__init__()
        self.num_mixer_layers = num_mixer_layers
        if (224 % patch_res) != 0:
            raise ValueError('size error')
        else:
            self.patch_res = patch_res
            self.n_patches = int((tf.square(224) / tf.square(self.patch_res)).numpy())
        self.hidden_size_c = hidden_size_c
        self.dim_dc = dim_dc
        self.dim_ds = dim_ds
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.stochastic_depth = stochastic_depth

        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.015),
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224)
        ])
        self.PatchConv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.hidden_size_c,
                                   kernel_size=(self.patch_res, self.patch_res),
                                   strides=(self.patch_res, self.patch_res),
                                   padding='valid',
                                   activation='linear'
                                   ),
            tf.keras.layers.Reshape((self.n_patches, self.hidden_size_c))
        ])
        survival_prob = 1 - tf.linspace(0., self.stochastic_depth, self.num_mixer_layers)
        self.Mixers = tf.keras.Sequential([
            MixerLayer(self.dim_dc,
                       self.dim_ds,
                       self.hidden_size_c,
                       self.n_patches,
                       self.dropout_rate,
                       survival_prob[i]
                       ) for i in range(self.num_mixer_layers)
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(self.num_labels,
                                  activation='softmax',
                                  kernel_initializer=tf.keras.initializers.zeros()
                                  )
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        if training:
            inputs = self.augmentation(inputs)
        patches = self.PatchConv(inputs)
        featuremap = self.Mixers(patches)
        y = self.classifier(featuremap)
        return y