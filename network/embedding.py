import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

class Embedding(object):

    def __init__(self, is_training, word_vec, word, pos1, pos2):
        temp_word_embedding = tf.get_variable(initializer=word_vec, name = 'temp_word_embedding', dtype=tf.float32)
        unk_word_embedding = tf.get_variable('unk_embedding', [FLAGS.word_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.word_vec = tf.concat([temp_word_embedding,
                                   tf.reshape(unk_word_embedding, [1, FLAGS.word_size]),
                                   tf.reshape(tf.constant(np.zeros([FLAGS.word_size], dtype=np.float32)), [1, FLAGS.word_size])], 0)
        self.word = word
        self.pos1 = pos1
        self.pos2 = pos2
        self.is_training = is_training

    def word_embedding(self, var_scope = None, reuse = False):
        with tf.variable_scope(var_scope or 'word_embedding', reuse = reuse):
            x = tf.nn.embedding_lookup(self.word_vec, self.word)
            return x

    def pos_embedding(self, simple_pos=False):
        with tf.name_scope("pos_embedding"):
            if simple_pos:
                temp_pos_array = np.zeros((FLAGS.pos_num + 1, FLAGS.pos_size), dtype=np.float32)
                temp_pos_array[(FLAGS.pos_num - 1) / 2] = np.ones(FLAGS.pos_size, dtype=np.float32)
                pos1_embedding = tf.constant(temp_pos_array)
                pos2_embedding = tf.constant(temp_pos_array)
            else:
                temp_pos1_embedding = tf.get_variable('temp_pos1_embedding', [FLAGS.pos_num, FLAGS.pos_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                pos1_embedding = tf.concat([temp_pos1_embedding, tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)), [1, FLAGS.pos_size])], 0)
                temp_pos2_embedding = tf.get_variable('temp_pos2_embedding', [FLAGS.pos_num, FLAGS.pos_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                pos2_embedding = tf.concat([temp_pos2_embedding, tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)), [1, FLAGS.pos_size])], 0)
            
            input_pos1 = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            input_pos2 = tf.nn.embedding_lookup(pos2_embedding, self.pos2)
            x = tf.concat(values = [input_pos1, input_pos2], axis = 2)
            return x

    def concat_embedding(self, word_embedding, pos_embedding):
        if pos_embedding is None:
            return word_embedding
        else:
            return tf.concat(values = [word_embedding, pos_embedding], axis = 2)
