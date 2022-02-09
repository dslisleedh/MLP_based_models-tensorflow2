import tensorflow as tf
import einops


class MultiScalePatchEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 output_channels,
                 m=[0, 1],
                 r=4):
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
                tf.keras.layers.Reshape((-1, k ** 2))
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
                 e = 4
                 ):
        super(FCBlock, self).__init__()
        self.d = d
        self.e = e

        self.forward = tf.keras.Sequential([

        ])


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
        self.vertical_fc = ###
        ])
        self.lnh = tf.keras.layers.LayerNormalization()
        self.horizon_fc = ###
        ])

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
    def __init__(self, c, e=4):
        super(ChannelMixingBlock, self).__init__()
        self.c = c
        self.e = e

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.c * self.e,
                                  activation='gelu',
                                  kernel_initializer='lecun_normal'
                                  ),
            tf.keras.layers.Dense(self.c,
                                  activation='linear'
                                  )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs) + inputs

