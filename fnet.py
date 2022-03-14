import tensorflow as tf
from einops import rearrange
from einops.layers.keras import Rearrange


class FourierTransformLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FourierTransformLayer, self).__init__()

        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        residual = tf.math.real(tf.signal.fft2d(tf.cast(inputs, 'complex64')))
        return self.ln(inputs + residual)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 dropout_rate,
                 expansion_rate
                 ):
        super(FeedForward, self).__init__()
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.expansion_rate = expansion_rate

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(self.expansion_rate * self.n_filters,
                                  activation='gelu'
                                  ),
            tf.keras.layers.Dense(self.n_filters)
        ])
        self.forward2 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward2(self.forward(inputs) + inputs)


class FBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 dropout_rate,
                 expansion_rate
                 ):
        super(FBlock, self).__init__()
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.expansion_rate = expansion_rate

        self.forward = tf.keras.Sequential([
            FourierTransformLayer(),
            FeedForward(self.n_filters, self.dropout_rate, self.expansion_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class FNetEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 n_layers,
                 dropout_rate,
                 expansion_rate=4
                 ):
        super(FNetEncoder, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.expansion_rate = expansion_rate

        self.forward = tf.keras.Sequential([
            FBlock(self.n_filters,
                   self.dropout_rate,
                   self.expansion_rate
                   ) for _ in range(self.n_layers)
        ])

    def call(self, inputs, training=None, mask=None):
        self.forward(inputs)


class FNet(tf.keras.models.Model):
    def __init__(self,

                 ):