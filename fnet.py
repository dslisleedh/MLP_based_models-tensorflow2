import tensorflow as tf


class FourierTransformLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FourierTransformLayer, self).__init__()

        self.ln = tf.keras.layers.LayerNormalization()

    @tf.function
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

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward2(self.forward(inputs) + inputs)


class FBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 dropout_rate,
                 expansion_rate,
                 ):
        super(FBlock, self).__init__()
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.expansion_rate = expansion_rate

        self.forward = tf.keras.Sequential([
            FourierTransformLayer(),
            FeedForward(self.n_filters, self.dropout_rate, self.expansion_rate)
        ])

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class FNetEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 n_filters,
                 n_layers,
                 dropout_rate,
                 expansion_rate
                 ):
        super(FNetEncoder, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.expansion_rate = expansion_rate

        dropouts = tf.linspace(0., self.dropout_rate, self.n_layers)
        self.forward = tf.keras.Sequential([
            FBlock(self.n_filters,
                   dropouts[i],
                   self.expansion_rate,
                   ) for i in range(self.n_layers)
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.forward(inputs)


class FNet(tf.keras.models.Model):
    def __init__(self,
                 patch_size=8,
                 n_filters=512,
                 n_layers=8,
                 dropout_rate=.25,
                 n_labels=3,
                 expansion_rate=4
                 ):
        super(FNet, self).__init__()
        self.patch_size = patch_size
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.n_labels = n_labels
        self.expansion_rate = expansion_rate

        self.patch_embedding = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.n_filters,
                                   activation='linear',
                                   kernel_size=self.patch_size,
                                   strides=self.patch_size,
                                   padding='VALID'
                                   ),
            tf.keras.layers.Reshape((-1, self.n_filters)),
        ])

        self.encoder = FNetEncoder(n_filters,
                                   n_layers,
                                   dropout_rate,
                                   expansion_rate
                                   )
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(self.n_labels,
                                  activation='softmax',
                                  kernel_initializer=tf.keras.initializers.zeros()
                                  )
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        patches = self.patch_embedding(inputs)
        featuremap = self.encoder(patches)
        y_hat = self.classifier(featuremap)
        return y_hat

