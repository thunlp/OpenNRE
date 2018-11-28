import tensorflow as tf
import numpy as np

def softmax_cross_entropy(x, label, rel_tot, weights_table=None, weights=1.0, var_scope=None):
    with tf.variable_scope(var_scope or "loss", reuse=tf.AUTO_REUSE):
        if weights_table is not None:
            weights = tf.nn.embedding_lookup(weights_table, label)
        label_onehot = tf.one_hot(indices=label, depth=rel_tot, dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x, weights=weights)
        tf.summary.scalar('loss', loss)
        return loss

def sigmoid_cross_entropy(x, label, rel_tot, weights_table=None, var_scope=None):
    with tf.variable_scope(var_scope or "loss", reuse=tf.AUTO_REUSE):
        if weights_table is None:
            weights = 1.0
        else:
            weights = tf.nn.embedding_lookup(weights_table, label)
        label_onehot = tf.one_hot(indices=label, depth=rel_tot, dtype=tf.int32)
        loss = tf.losses.sigmoid_cross_entropy(label_onehot, logits=x, weights=weights)
        tf.summary.scalar('loss', loss)
        return loss

# Soft-label
# I just implemented it, but I haven't got the result in paper.
def soft_label_softmax_cross_entropy(x):
    with tf.name_scope("soft-label-loss"):
        label_onehot = tf.one_hot(indices=self.label, depth=FLAGS.num_classes, dtype=tf.int32)
        nscore = x + 0.9 * tf.reshape(tf.reduce_max(x, 1), [-1, 1]) * tf.cast(label_onehot, tf.float32)
        nlabel = tf.one_hot(indices=tf.reshape(tf.argmax(nscore, axis=1), [-1]), depth=FLAGS.num_classes, dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=nlabel, logits=nscore, weights=self.weights)
        tf.summary.scalar('loss', loss)
        return loss

def output(x):
    return tf.argmax(x, axis=-1)
