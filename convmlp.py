from utils import Droppath
from einops.layers.keras import Rearrange
import tensorflow as tf


class ConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(ConvolutionalBlock, self).__init__()
        self.filters = filters
        self.strides = strides

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.filters,
                                   kernel_size=(3, 3),
                                   kernel_initializer='he_normal',
                                   strides=(self.strides, self.strides),
                                   padding='same',
                                   use_bias=False
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class ConvolutionalTokenizer(tf.keras.layers.Layer):
    def __init__(self, filters=64):
        super(ConvolutionalTokenizer, self).__init__()
        self.filters = filters

        self.forward = tf.keras.Sequential([
            ConvolutionalBlock(self.filters if i == 2 else self.filters // 2,
                               2 if i == 0 else 1
                               ) for i in range(3)
        ] + [
            tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=(2, 2),
                                      padding='valid'
                                      )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class PureConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters_in,
                 filters_hidden
                 ):
        super(PureConvBlock, self).__init__()
        self.filters_in = filters_in
        self.filters_hidden = filters_hidden

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filters_hidden,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='valid',
                                   use_bias=False
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.filters_hidden,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.filters_in,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='valid',
                                   use_bias=False
                                   )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class ConvStage(tf.keras.layers.Layer):
    def __init__(self,
                 filters_in=64,
                 filters_hidden=128,
                 filters_out=128,
                 n_blocks=2
                 ):
        super(ConvStage, self).__init__()
        self.filters_in = filters_in
        self.filters_hidden = filters_hidden
        self.filters_out = filters_out
        self.n_blocks = n_blocks

        self.forward = tf.keras.Sequential([
            PureConvBlock(self.filters_in,
                          self.filters_hidden
                          ) for _ in range(self.n_blocks)
        ] + [
            tf.keras.layers.Conv2D(self.filters_out,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding='same'
                                   )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class ChannelMlp(tf.keras.layers.Layer):
    def __init__(self,
                 filters_in,
                 filters_hidden
                 ):
        super(ChannelMlp, self).__init__()
        self.filters_in = filters_in
        self.filters_hidden = filters_hidden

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(filters_hidden,
                                  activation='gelu'
                                  ),
            tf.keras.layers.Dense(filters_in)
        ])
        
    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class ConvMlpBlock(tf.keras.layers.Layer):
    def __init__(self,
                 dim_embedding,
                 filters_hidden,
                 survival_prob
                 ):
        super(ConvMlpBlock, self).__init__()
        self.dim_embedding = dim_embedding
        self.filters_hidden = filters_hidden
        self.survival_prob = survival_prob

        self.mlp1 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            ChannelMlp(self.dim_embedding,
                       self.filters_hidden
                       )
        ])
        if self.survival_prob != 1.:
            self.mlp1.add(Droppath(survival_prob))

        self.dwconv = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='linear',
                                            use_bias=False
                                            )
        ])
        if self.survival_prob != 1.:
            self.dwconv.add(Droppath(survival_prob))

        self.mlp2 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            ChannelMlp(self.dim_embedding,
                       self.filters_hidden
                       )
        ])
        if self.survival_prob != 1.:
            self.mlp2.add(Droppath(survival_prob))

    def call(self, inputs, **kwargs):
        return self.mlp2(self.dwconv(self.mlp1(inputs)))


class ConvMlpStage(tf.keras.layers.Layer):
    def __init__(self,
                 n_blocks,
                 filters_embedding,
                 r,
                 stochastic_depth,
                 downsample=True
                 ):
        super(ConvMlpStage, self).__init__()
        self.n_blocks = n_blocks
        self.filters_embedding = filters_embedding
        self.r = r
        self.stochastic_depth = stochastic_depth
        self.survival_prob = 1. - tf.linspace(0., self.stochastic_depth, self.n_blocks)
        self.downsmaple = downsample

        self.forward = tf.keras.Sequential([
            ConvMlpBlock(self.filters_embedding,
                         self.filters_embedding * r,
                         survival_prob
                         ) for survival_prob in self.survival_prob
        ])
        if self.downsmaple:
            self.forward.add(tf.keras.layers.Conv2D(filters=filters_embedding,
                                                    kernel_size=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same'
                                                    )
                             )

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class ConvMlp(tf.keras.models.Model):
    def __init__(self,
                 blocks,
                 channels,
                 r,
                 n_conv_blocks,
                 token_channels=64,
                 stochastic_depth=.1,
                 num_classes=1000
                 ):
        super(ConvMlp, self).__init__()
        if len(blocks) != len(channels):
            raise ValueError('Size error')
        else:
            self.blocks = blocks
            self.channels = channels
        self.r = r
        self.n_conv_blocks = n_conv_blocks
        self.token_channels = token_channels
        self.stochastic_depth = stochastic_depth
        self.num_classes = num_classes

        self.tokenizer = ConvolutionalTokenizer(filters=token_channels)

        self.stages = tf.keras.Sequential([
            ConvStage(filters_in=self.token_channels,
                      filters_hidden=self.channels[0],
                      filters_out=self.channels[0],
                      n_blocks=self.n_conv_blocks
                      )
        ])
        for i in range(len(self.blocks)):
            self.stages.add(ConvMlpStage(self.blocks[i],
                                         self.channels[i],
                                         self.r,
                                         self.stochastic_depth,
                                         downsample=True if (i+1) == len(self.blocks) else False
                                         )
                            )

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            Rearrange('B H W C -> B (H W) C'),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(self.num_classes,
                                  activation='softmax',
                                  kernel_initializer=tf.keras.initializers.zeros()
                                  )
        ])

    def call(self, inputs, training=None, mask=None):
        tokens = self.tokenizer(inputs)
        featuremap = self.stages(tokens)
        y = self.classifier(featuremap)
        return y