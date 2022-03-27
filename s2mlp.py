from utils import Droppath
import tensorflow as tf


class SpatialShift(tf.keras.layers.Layer):
    def __init__(self, g=4):
        super(SpatialShift, self).__init__()
        self.g = g

    @tf.function
    def spatial_shift(self, x):
        b, h, w, c = x.get_shape().as_list()
        return tf.concat([tf.concat([x[:, :1, :, :c // 4], x[:, 0:h - 1, :, :c // 4]],
                                    axis=1
                                    ),
                          tf.concat([x[:, 1:h, :, c // 4:c // 2], x[:, h - 1:, :, c // 4:c // 2]],
                                    axis=1
                                    ),
                          tf.concat([x[:, :, :1, c // 2:3 * (c // 4)], x[:, :, 0:w - 1, c // 2:3 * (c // 4)]],
                                    axis=2
                                    ),
                          tf.concat([x[:, :, 1:w, 3 * (c // 4):], x[:, :, w - 1:, 3 * (c // 4):]],
                                    axis=2
                                    )
                          ],
                         axis=-1
                         )

    def call(self, inputs, **kwargs):
        groups = tf.split(inputs,
                          axis=-1,
                          num_or_size_splits=self.g
                          )
        return tf.concat([self.spatial_shift(g) for g in groups],
                         axis=-1
                         )


class MLP(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 r,
                 shift=True
                 ):
        super(MLP, self).__init__()
        self.c = c
        self.r = r
        self.shift = shift

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.c * self.r,
                                  activation='gelu'
                                  )
        ])
        if self.shift:
            self.forward.add(SpatialShift())
        self.forward.add(tf.keras.layers.Dense(c,
                                               activation='linear'
                                               )
                         )
        self.forward.add(tf.keras.layers.LayerNormalization())

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class S2Block(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 r,
                 survival_prob
                 ):
        super(S2Block, self).__init__()
        self.c = c
        self.r = r
        self.survival_prob = survival_prob

        self.spatial = MLP(self.c,
                           self.r
                           )
        self.channel = MLP(self.c,
                           self.r,
                           shift=False
                           )
        self.stochastic_depth = Droppath(self.survival_prob)

    def call(self, inputs, **kwargs):
        inputs = self.stochastic_depth(self.spatial(inputs)) + inputs
        return self.stochastic_depth(self.channel(inputs)) + inputs


class S2Mlp(tf.keras.models.Model):
    def __init__(self,
                 p,
                 c,
                 r,
                 n,
                 num_classes,
                 stochastic_depth=.1
                 ):
        '''
        Train input_size : 256
        Test/Inference input_size : 224
        :param p: patch size
        :param c: channels
        :param r: expansion rate
        :param n: number of s2blocks
        :param stochastic_depth: stochastic depth rate
        '''
        super(S2Mlp, self).__init__()
        self.p = p
        self.c = c
        self.r = r
        self.n = n
        self.num_classes = num_classes
        self.stochastic_depth = stochastic_depth

        self.PatchConv = tf.keras.layers.Conv2D(filters=3*self.p*self.p,
                                                kernel_size=self.p,
                                                activation='linear',
                                                padding='valid',
                                                strides=self.p
                                                )
        survival_prob = 1 - tf.linspace(0., self.stochastic_depth, self.n)
        self.Blocks = tf.keras.Sequential([
           S2Block(self.c,
                   self.r,
                   survival_prob=survival_prob[i]
                   ) for i in range(self.n)
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(self.num_classes,
                                  activation='softmax'
                                  )
        ])

    def call(self, inputs, training=None, mask=None):
        patches = self.PatchConv(inputs)
        featuremap = self.Blocks(patches)
        y = self.classifier(featuremap)
        return y