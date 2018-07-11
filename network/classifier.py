import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

class Classifier(object):

    def __init__(self, is_training, label, weights):
        self.label = label
        self.weights = weights
        self.is_training = is_training

    def softmax_cross_entropy(self, x):
        print x
        print self.label.shape
        print FLAGS.num_classes
        with tf.name_scope("loss"):
            label_onehot = tf.one_hot(indices=self.label, depth=FLAGS.num_classes, dtype=tf.int32)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x, weights=self.weights)
            #loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x)
            tf.summary.scalar('loss', loss)
            return loss

    # Soft-label
    # I just implemented it, but I haven't got the result in paper.
    def soft_label_softmax_cross_entropy(self, x):
        with tf.name_scope("soft-label-loss"):
            label_onehot = tf.one_hot(indices=self.label, depth=FLAGS.num_classes, dtype=tf.int32)
            nscore = x + 0.9 * tf.reshape(tf.reduce_max(x, 1), [-1, 1]) * tf.cast(label_onehot, tf.float32)
            nlabel = tf.one_hot(indices=tf.reshape(tf.argmax(nscore, axis=1), [-1]), depth=FLAGS.num_classes, dtype=tf.int32)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=nlabel, logits=nscore, weights=self.weights)
            tf.summary.scalar('loss', loss)
            return loss

    def output(self, x):
        with tf.name_scope("output"):
            output = tf.argmax(x, 1, name="output")
            return output
