from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, EmbeddingWrapper
import tensorflow as tf


def loss():
    


class Model(object):
    def __init__(self, vocabulary_size, layers=(256, 256,)):
        self.x = tf.placeholder(tf.int32, shape=[None])
        self.v = vocabulary_size

        embedding = EmbeddingWrapper(cell=BasicLSTMCell, embedding_classes=vocabulary_size, embedding_size=64)
        embedded = embedding(self.x)
        network_cells = [
            BasicLSTMCell(num_units=layer_size) for layer_size in layers
        ]
        network = MultiRNNCell(cells=network_cells)
        self.out = network(embedded)

    def step(self):
        pass

    def probability(self, sequence):
        pass

    def generate(self):
        pass


if __name__ == '__main__':
    with tf.Session() as sess:
        pass