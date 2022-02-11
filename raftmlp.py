import tensorflow as tf
import einops
from utils import Droppath
from einops.layers.keras import Rearrange


class MultiScalePatchEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 r,
                 output_channels,
                 m=[0, 1]
                 ):
        super(MultiScalePatchEmbedding, self).__init__()
        self.output_channels = output_channels
        self.strides = r
        self.kernels = [r*(2**i) for i in m]

        self.patch_embedding = [
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=k ** 2 if len(self.kernels) != 1 else self.output_channels,
                                       kernel_size=k,
                                       strides=self.strides,
                                       padding='same',
                                       ),
                Rearrange('B H W C -> B (H W) C')
            ]) for k in self.kernels
        ]
        if len(self.kernels) == 1:
            self.fc = tf.keras.layers.Layer()
        else:
            self.fc = tf.keras.layers.Dense(self.output_channels,
                                            activation='linear'
                                            )

    def call(self, inputs, **kwargs):
        output = []
        for emb in self.patch_embedding:
            output.append(emb(inputs))
        return self.fc(tf.concat(output,
                                 axis=-1)
                       )


class FCBlock(tf.keras.layers.Layer):
    def __init__(self,
                 d,
                 e=4
                 ):
        super(FCBlock, self).__init__()
        self.d = d
        self.e = e

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d * self.e,
                                  activation='gelu',
                                  kernel_initializer='lecun_normal'
                                  ),
            tf.keras.layers.Dense(self.d,
                                  activation='linear'
                                  )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class RaftTokenMixingBlock(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 h,
                 w,
                 survival_prob,
                 e=2,
                 r=2,
                 ):
        super(RaftTokenMixingBlock, self).__init__()
        self.r = r
        self.c = c
        self.o = c//r
        self.h = h
        self.w = w
        self.survival_prob = survival_prob
        self.e = e

        self.lnv = tf.keras.layers.LayerNormalization()
        self.vertical_fc = FCBlock(self.r * self.h,
                                   e=self.e
                                   )
        self.lnh = tf.keras.layers.LayerNormalization()
        self.horizon_fc = FCBlock(self.r * self.w,
                                  e=self.e
                                  )
        self.droppath = Droppath(self.survival_prob) if self.survival_prob != 1. else tf.keras.layers.Layer()

    def call(self, inputs, **kwargs):
        y = self.lnv(inputs)
        y = einops.rearrange(y, 'b (h w) (r o) -> b (o w) (r h)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        y = self.vertical_fc(y)
        y = einops.rearrange(y, 'b (o w) (r h) -> b (h w) (r o)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        inputs = self.droppath(y) + inputs
        y = self.lnh(inputs)
        y = einops.rearrange(y, 'b (h w) (r o) -> b (o h) (r w)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        y = self.horizon_fc(y)
        y = einops.rearrange(y, 'b (o h) (r w) -> b (h w) (r o)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        return self.droppath(y) + inputs


class ChannelMixingBlock(tf.keras.layers.Layer):
    def __init__(self, c, survival_prob):
        super(ChannelMixingBlock, self).__init__()
        self.c = c
        self.survival_prob = survival_prob

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            FCBlock(self.c)
        ])

    def call(self, inputs, **kwargs):
        stochastic_depth = tf.keras.backend.random_bernoulli(shape=(1,),
                                                             p=self.survival_prob
                                                             )
        return (self.forward(inputs) * stochastic_depth) + inputs


class RaftBlock(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 h,
                 w,
                 survival_prob
                 ):
        super(RaftBlock, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.survival_prob = survival_prob

        self.forward = tf.keras.Sequential([
            RaftTokenMixingBlock(self.c, self.h, self.w, self.survival_prob),
            ChannelMixingBlock(self.c, self.survival_prob)
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Level(tf.keras.layers.Layer):
    def __init__(self,
                 layer,
                 h,
                 w,
                 c,
                 num_raftblocks,
                 survival_prob
                 ):
        super(Level, self).__init__()
        self.layer = layer
        self.r = 4 if self.layer == 1 else 2
        self.h = h
        self.w = w
        self.c = c
        self.num_raftblocks = num_raftblocks
        self.survival_prob = survival_prob

        self.forward = tf.keras.Sequential([
            MultiScalePatchEmbedding(self.r, self.c, m=[0] if self.layer == 4 else [0, 1])
        ] + [
            RaftBlock(self.c,
                      self.h,
                      self.w,
                      self.survival_prob
                      ) for _ in range(num_raftblocks)
        ] + [
            Rearrange('b (h w) c -> b h w c',
                      h=self.h, w=self.w
                      )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class RaftMlp(tf.keras.models.Model):
    '''
    train input size : 256
    test/inference input size : 224
    '''
    def __init__(self,
                 num_blocks,
                 num_channels,
                 num_classes,
                 input_size=224,
                 stochastic_depth=.1
                 ):
        super(RaftMlp, self).__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.stochastic_depth = stochastic_depth
        self.survival_prob = 1. - tf.linspace(0., self.stochastic_depth, 4)

        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.015),
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=self.input_size, width=self.input_size)
        ])
        self.levels = tf.keras.Sequential([
            Level(layer=i+1,
                  h=self.input_size // (2 ** (i + 2)),
                  w=self.input_size // (2 ** (i + 2)),
                  c=self.num_channels[i],
                  num_raftblocks=self.num_blocks[i],
                  survival_prob=self.survival_prob[i]
                  ) for i in range(4)
        ])

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(self.num_classes,
                                  activation='softmax',
                                  kernel_initializer=tf.keras.initializers.zeros()
                                  )
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        if training:
            inputs = self.augmentation(inputs)
        featuremap = self.levels(inputs)
        y = self.classifier(featuremap)
        return y