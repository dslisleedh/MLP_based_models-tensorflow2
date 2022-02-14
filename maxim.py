from utils import Droppath
from einops.layers.keras import Rearrange
import tensorflow as tf


'''
incomplete
'''


class MabGmlpBlock(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 n_patches,
                 e=6
                 ):
        super(MabGmlpBlock, self).__init__()
        self.c = c
        self.n_patches = n_patches
        self.e = e

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LocallyConnected1D(filters=self.c * self.e,
                                               kernel_size=1,
                                               strides=1,
                                               padding='valid',
                                               activation='gelu'
                                               )
        ])
        self.sgu_res = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Permute((2, 1)),
            tf.keras.layers.LocallyConnected1D(filters=self.n_patches,
                                               kernel_size=1,
                                               strides=1,
                                               padding='valid',
                                               activation='linear',
                                               kernel_initializer=tf.keras.initializers.truncated_normal(stddev=.001),
                                               bias_initializer=tf.keras.initializers.ones()
                                               ),
            tf.keras.layers.Permute((2, 1))
        ])
        self.dense = tf.keras.layers.LocallyConnected1D(filters=self.c,
                                                        kernel_size=1,
                                                        strides=1,
                                                        padding='valid',
                                                        activation='linear'
                                                        )

    def call(self, inputs, **kwargs):
        b, _, n, c = tf.shape(inputs)
        y = tf.reshape(inputs, [b, n, c])
        y = self.forward(y)
        u, v = tf.split(y, axis=-1, num_or_size_splits=2)
        u = self.sgu_res(u)
        y = self.dense(u * v)
        return tf.reshape(y, [b, 1, n, c]) + inputs


class Mab(tf.keras.layers.Layer):
    def __init__(self,
                 h,
                 w,
                 c,
                 split_size,
                 e=6
                 ):
        super(Mab, self).__init__()
        self.h = h
        self.w = w
        self.c = c
        self.b = self.d = split_size
        self.n_patches = split_size ** 2
        self.n_heads = (self.h//split_size) * (self.w//split_size)
        self.e = e

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.c * self.e,
                                  activation='gelu',
                                  kernel_initializer='lecun_normal'
                                  )
        ])
        self._set_gridsblocks_makers(self.h,
                                     self.w,
                                     self.b,
                                     self.d
                                     )
        self.gmlp = MabGmlpBlock(self.c//2,
                                 self.n_patches
                                 )
        self.dense = tf.keras.layers.Dense(self.c,
                                           activation='linear'
                                           )

    def _set_gridsblocks_makers(self, h, w, b, d):
        self.make_blocks = Rearrange('B (H b1) (W b2) C -> B (H W) (b1 b2) C',
                                     H=h // b, W=w // b)
        self.recon_blocks = Rearrange('B (H W) (b1 b2) C -> B (H b1) (W b2) C',
                                      H=w // b, W=w // b, b1=b, b2=b)
        self.make_grids = Rearrange('B (d1 H) (d2 W) C -> B (H W) (d1 d2) C',
                                    H=h // d, W=w // d)
        self.recon_grids = Rearrange('B (H W) (d1 d2) C -> B (d1 H) (d2 W) C',
                                     H=h // d, W=w // d, d1=d, d2=d)

    def call(self, inputs, **kwargs):
        y = self.forward(inputs)
        local_branch, global_branch = tf.split(y, axis=-1, num_or_size_splits=2)
        local_branch = tf.split(self.make_blocks(local_branch), axis=1, num_or_size_splits=self.n_heads)
        local_branch = tf.concat([self.gmlp(lb) for lb in local_branch], axis=1)
        local_branch = self.recon_grids(local_branch)
        global_branch = tf.split(self.make_blocks(global_branch), axis=1, num_or_size_splits=self.n_heads)
        global_branch = tf.concat([self.gmlp(gb) for gb in global_branch], axis=1)
        global_branch = self.recon_blocks(global_branch)
        y = tf.concat([local_branch, global_branch], axis=-1)
        return y + inputs, global_branch


class Rcab(tf.keras.layers.Layer):
    def __init__(self,
                 c,
                 kernel_size,
                 r=4
                 ):
        '''
        https://arxiv.org/pdf/1807.06521.pdf
        :param c:
        :param kernel_size:
        :param r:
        '''
        super(Rcab, self).__init__()
        self.c = c
        self.kernel_size = kernel_size
        self.r = r

        self.Conv = tf.keras.layers.Conv2D(self.c,
                                           kernel_size=self.kernel_size,
                                           activation='linear',
                                           padding='same',
                                           strides=1,
                                           )
        self.MP = tf.keras.layers.GlobalMaxPool2D()
        self.AP = tf.keras.layers.GlobalAvgPool2D()

        self.MLP = tf.keras.Sequential([
            tf.keras.layers.Dense(self.c // self.r,
                                  activation='relu'
                                  ),
            tf.keras.layers.Dense(self.c,
                                  activation='relu'
                                  )
        ])
        self.DepthMP = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x,
                                                                      axis=-1,
                                                                      keepdims=True
                                                                      )
                                              )
        self.DepthAP = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x,
                                                                       axis=-1,
                                                                       keepdims=True
                                                                       )
                                              )
        self.SpatialConv = tf.keras.layers.Conv2D(filters=1,
                                                  kernel_size=7,
                                                  strides=1,
                                                  padding='same',
                                                  use_bias=False,
                                                  activation='sigmoid'
                                                  )

    def call(self, inputs, **kwargs):
        f = self.Conv(inputs)
        cmp = self.MLP(self.MP(f))
        cap = self.MLP(self.AP(f))
        b, c = tf.shape(cap)
        channel_attention = tf.reshape(tf.nn.sigmoid(cmp + cap), (b, 1, 1, c))
        f_ = tf.multiply(f, channel_attention)
        f_s = tf.concat([self.DepthMP(f_), self.DepthAP(f_)],
                        axis=-1
                        )
        spatial_attention = self.SpatialConv(f_s)
        f__ = tf.multiply(f_, spatial_attention)
        return inputs + f__

