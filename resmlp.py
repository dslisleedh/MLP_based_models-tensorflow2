import tensorflow as tf


class PreAffine(tf.keras.layers.Layer):
    def __init__(self, dims):
        super(PreAffine, self).__init__()
        self.dims = dims

        self.alpha = tf.Variable(tf.ones((1, 1, self.dims)))
        self.beta = tf.Variable(tf.zeros((1, 1, self.dims)))

    def call(self, inputs, **kwargs):
        return self.alpha * inputs + self.beta


class PostAffine(tf.keras.layers.Layer):
    '''
    Works same as LayerScale from https://arxiv.org/abs/2103.17239v2
    '''
    def __init__(self,
                 dim,
                 depth
                 ):
        super(PostAffine, self).__init__()
        self.dim = dim
        self.depth = depth
        if self.depth < 18:
            self.epsilon = .1
        elif self.depth < 24:
            self.epsilon = 1e-5
        else:
            self.epsilon = 1e-6

        self.alpha = tf.Variable(tf.fill((1, 1, self.dim), self.epsilon),
                                 trainable=True,
                                 dtype='float32'
                                 )

    def call(self, inputs, **kwargs):
        return self.alpha * inputs


class Mlp(tf.keras.layers.Layer):
    def __init__(self, dims):
        super(Mlp, self).__init__()
        self.dims = dims

        self.fc1 = tf.keras.layers.Dense(4 * self.dims,
                                         activation='linear',
                                         kernel_initializer='lecun_normal'
                                         )
        self.fc2 = tf.keras.layers.Dense(self.dims,
                                         activation='linear',
                                         kernel_initializer='lecun_normal'
                                         )

    def call(self, inputs, **kwargs):
        return self.fc2(tf.nn.gelu(self.fc1(inputs)))


class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_patches,
                 n_dims,
                 depth,
                 survival_prob
                 ):
        super(ResBlock, self).__init__()
        self.n_patches = n_patches
        self.n_dims = n_dims
        self.depth = depth
        self.survival_prob = survival_prob

        self.Cross_patch = tf.keras.Sequential([
            PreAffine(self.n_dims),
            tf.keras.layers.Permute((2, 1)),
            tf.keras.layers.Dense(self.n_patches,
                                  activation='linear'
                                  ),
            tf.keras.layers.Permute((2, 1)),
            PostAffine(self.n_dims, self.depth)
        ])
        self.Cross_channel = tf.keras.Sequential([
            PreAffine(self.n_dims),
            Mlp(self.n_dims),
            PostAffine(self.n_dims, self.depth)
        ])

    def call(self, inputs, **kwargs):
        stochastic_depth = tf.keras.backend.random_bernoulli(shape=(1,),
                                                             p=self.survival_prob
                                                             )
        z = self.Cross_patch(inputs) * stochastic_depth + inputs
        y = self.Cross_channel(z) * stochastic_depth + z
        return y


class ResMlp(tf.keras.models.Model):
    '''
    Input image resolution : 256
    there's no mention specific rate of regularization in paper
    '''
    def __init__(self,
                 patch_res,
                 n_layers,
                 dims,
                 n_labels,
                 stochastic_depth_rate=.1
                 ):
        super(ResMlp, self).__init__()
        if (224 % patch_res) != 0:
            raise ValueError('size error')
        else:
            self.patch_res = patch_res
            self.n_patches = int((tf.square(224) / tf.square(self.patch_res)).numpy())
        self.n_layers = n_layers
        self.dims = dims
        self.n_labels = n_labels
        self.stochastic_depth_rate = stochastic_depth_rate

        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=.015),
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224)
        ])
        self.patch_projector = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.dims,
                                   kernel_size=(self.patch_res, self.patch_res),
                                   strides=(self.patch_res, self.patch_res),
                                   padding='valid',
                                   activation='linear'
                                   ),
            tf.keras.layers.Reshape((self.n_patches, self.dims))
        ])
        survival_prob = 1 - tf.linspace(0., self.stochastic_depth_rate, self.n_layers)
        self.blocks = tf.keras.Sequential([
            ResBlock(self.n_patches,
                     self.dims,
                     i + 1,
                     survival_prob[i]
                     ) for i in range(self.n_layers)
        ])
        self.classifier = tf.keras.Sequential([
            PreAffine(self.dims),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(self.n_labels,
                                  activation='softmax',
                                  kernel_initializer=tf.keras.initializers.zeros()
                                  )
        ])

    @tf.function
    def call(self, inputs, training=None, mask=None):
        augs = self.augmentation(inputs)
        patches = self.patch_projector(augs)
        featuremap = self.blocks(patches)
        y = self.classifier(featuremap)
        return y