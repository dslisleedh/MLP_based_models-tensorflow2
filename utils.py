import tensorflow as tf


class Droppath(tf.keras.layers.Layer):
    def __init__(self, prob):
        super(Droppath, self).__init__()
        self.prob = prob

    @tf.function
    def call(self, inputs, **kwargs):
        if tf.keras.backend.learning_phase():
            epsilon = tf.keras.backend.random_bernoulli(shape=[tf.shape(inputs)[0]] + [1 for _ in range(len(tf.shape(inputs)) - 1)],
                                                        p=self.prob,
                                                        dtype='float32'
                                                        )
            return inputs * epsilon
        else:
            return inputs * self.prob

