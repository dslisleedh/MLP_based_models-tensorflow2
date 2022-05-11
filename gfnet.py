import tensorflow as tf
import einops
from utils import Droppath
'''
only implemented vanilla GFNet
No Layer scale applied in vanilla GFNet
'''


class GlobalFilterLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GlobalFilterLayer, self).__init__()
        self.ln = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        b, h, w, c = input_shape
        self.real = self.add_weight(name='real',
                                    shape=[1, c, h, (w//2)+1],
                                    initializer=tf.keras.initializers.truncated_normal(mean=0, stddev=.2),
                                    trainable=True
                                    )
        self.imag = self.add_weight(name='imag',
                                    shape=[1, c, h, (w // 2) + 1],
                                    initializer=tf.keras.initializers.truncated_normal(mean=0, stddev=.2),
                                    trainable=True
                                    )
        super(GlobalFilterLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        inputs = einops.rearrange(inputs, 'b h w c -> b c h w')
        inputs_complex = tf.signal.rfft2d(inputs)
        k = tf.complex(self.real, self.imag)
        inputs_complex = inputs_complex * k
        inputs = tf.signal.irfft2d(inputs_complex)
        inputs = einops.rearrange(inputs, 'b c h w -> b h w c')
        return inputs


class FFN(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 expansion_rate=4
                 ):
        super(FFN, self).__init__()
        self.n_filters = n_filters
        self.expansion_rate = expansion_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.n_filters * self.expansion_rate),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.n_filters)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class GFBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 survival_prob
                 ):
        super(GFBlock, self).__init__()
        self.n_filters = n_filters
        self.survival_prob = survival_prob

        self.forward = tf.keras.Sequential([
            GlobalFilterLayer(),
            FFN(self.n_filters)
        ])
        self.stochastic_depth = Droppath(self.survival_prob)

    def call(self, inputs, *args, **kwargs):
        return self.stochastic_depth(self.forward(inputs)) + inputs


class GFNet(tf.keras.models.Model):
    def __init__(self,
                 input_res: int = 224,
                 patch_size: int = 16,
                 n_filters: int = 768,
                 n_layers: int = 12,
                 uniform_drop: bool = False,
                 stochastic_depth_rate: float = 0.,
                 n_labels: int = 1000
                 ):
        super(GFNet, self).__init__()
        if (input_res % patch_size) != 0:
            raise ValueError('size error')
        else:
            self.input_res = input_res
            self.patch_size = patch_size
            self.n_patches = int((tf.square(self.input_res) / tf.square(self.patch_size)).numpy())
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.stochastic_depth_rate = stochastic_depth_rate
        self.uniform_drop = uniform_drop
        if self.uniform_drop:
            self.survival_prob = 1. - tf.convert_to_tensor([self.stochastic_depth_rate for _ in range(self.n_layers)])
        else:
            self.survival_prob = 1. - tf.linspace(0., self.stochastic_depth_rate, self.n_layers)
        self.n_labels = n_labels

        self.patch_embedding = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                                   self.n_filters,
                                   kernel_size=self.patch_size,
                                   strides=self.patch_size,
                                   padding='VALID',
                                   activation='linear',
                                   use_bias=False
                                   ),
            tf.keras.layers.Reshape((-1, self.n_filters))
        ])
        self.pos_embedding = tf.Variable(
            tf.random.truncated_normal(shape=(1, self.n_patches, self.n_filters),
                                       mean=0,
                                       stddev=.2
                                       ),
            trainable=True
        )
        self.blocks = tf.keras.Sequential([
            GFBlock(self.n_filters, survival_prob) for survival_prob in self.survival_prob
        ] + [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GlobalAvgPool2D()
        ])
        self.classifier = tf.keras.layers.Dense(self.n_labels,
                                                activation='softmax',
                                                kernel_initializer=tf.keras.initializers.zeros()
                                                )

    def call(self, inputs, training=None, mask=None):
        _, h, w, _ = inputs.shape
        embedding = self.patch_embedding(inputs) + self.pos_embedding
        embedding = einops.rearrange(embedding, 'b (h_p w_p) c -> b h_p w_p c',
                                     h_p=h//self.patch_size, w_p=w//self.patch_size
                                     )
        features = self.blocks(embedding)
        y_hat = self.classifier(features)
        return y_hat
