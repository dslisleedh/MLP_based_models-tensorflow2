import tensorflow as tf
import einops
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
                 e=2,
                 r=2
                 ):
        super(RaftTokenMixingBlock, self).__init__()
        self.r = r
        self.c = c
        self.o = c//r
        self.h = h
        self.w = w
        self.e = e

        self.lnv = tf.keras.layers.LayerNormalization()
        self.vertical_fc = FCBlock(self.r * self.h,
                                   e=self.e
                                   )
        self.lnh = tf.keras.layers.LayerNormalization()
        self.horizon_fc = FCBlock(self.r * self.w,
                                  e=self.e
                                  )

    def call(self, inputs, **kwargs):
        y = self.lnv(inputs)
        y = einops.rearrange(y, 'b (h w) (r o) -> b (o w) (r h)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        y = self.vertical_fc(y)
        y = einops.rearrange(y, 'b (o w) (r h) -> b (h w) (r o)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        inputs = y + inputs
        y = self.lnh(inputs)
        y = einops.rearrange(y, 'b (h w) (r o) -> b (o h) (r w)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        y = self.horizon_fc(y)
        y = einops.rearrange(y, 'b (o h) (r w) -> b (h w) (r o)',
                             h=self.h, w=self.w, r=self.r, o=self.o
                             )
        return y + inputs


class ChannelMixingBlock(tf.keras.layers.Layer):
    def __init__(self, c):
        super(ChannelMixingBlock, self).__init__()
        self.c = c

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            FCBlock(self.c)
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs) + inputs


class RaftBlock(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 h,
                 w
                 ):
        super(RaftBlock, self).__init__()
        self.c = c
        self.h = h
        self.w = w

        self.forward = tf.keras.Sequential([
            RaftTokenMixingBlock(self.c, self.h, self.w),
            ChannelMixingBlock(self.c)
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
                 ):
        super(Level, self).__init__()
        self.layer = layer
        self.r = 4 if self.layer == 1 else 2
        self.h = h
        self.w = w
        self.c = c
        self.num_raftblocks = num_raftblocks

        self.forward = tf.keras.Sequential([
            MultiScalePatchEmbedding(self.r, self.c, m=[0] if self.layer == 3 else [0, 1])
        ] + [
            RaftBlock(self.c,
                      self.h,
                      self.w) for _ in range(num_raftblocks)
        ] + [
            Rearrange('b (h w) c -> b h w c',
                      h=self.h, w=self.w
                      )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class RaftMlp(tf.keras.models.Model):
    def __init__(self,
                 num_blocks,
                 num_channels,
                 num_classes,
                 input_size=224
                 ):
        super(RaftMlp, self).__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_size = input_size

        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.015),
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224)
        ])
        self.levels = tf.keras.Sequential([
            Level(layer=i+1,
                  h=self.input_size // (2 ** (i + 2)),
                  w=self.input_size // (2 ** (i + 2)),
                  c=self.num_channels[i],
                  num_raftblocks=self.num_blocks[i]
                  ) for i in range(4)
        ])

        self.classifier = tf.keras.Sequential([
            Rearrange('b h w c -> b (h w) c'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(self.num_classes,
                                  activation='softmax'
                                  )
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        if training:
            inputs = self.augmentation(inputs)
        featuremap = self.levels(inputs)
        y = self.classifier(featuremap)
        return y