import tensorflow as tf
import numpy as np

def word_embedding(word, word_vec_mat, var_scope=None, word_embedding_dim=50):
    with tf.variable_scope(var_scope or 'word_embedding', reuse=tf.AUTO_REUSE):
        word_embedding = tf.get_variable('word_embedding', initializer=word_vec_mat, dtype=tf.float32)
        x = tf.nn.embedding_lookup(word_embedding, word)
        return x

def pos_embedding(pos1, pos2, var_scope=None, pos_embedding_dim=5, max_length=120):
    with tf.variable_scope(var_scope or 'pos_embedding', reuse=tf.AUTO_REUSE):
        pos_tot = max_length * 2

        real_pos1_embedding = tf.get_variable('real_pos1_embedding', [pos_tot - 1, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
        pos1_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), real_pos1_embedding], 0)
        real_pos2_embedding = tf.get_variable('real_pos2_embedding', [pos_tot - 1, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
        pos2_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), real_pos2_embedding], 0)

        input_pos1 = tf.nn.embedding_lookup(pos1_embedding, pos1)
        input_pos2 = tf.nn.embedding_lookup(pos2_embedding, pos2)
        x = tf.concat([input_pos1, input_pos2], -1)
        return x

def word_position_embedding(word, word_vec_mat, pos1, pos2, var_scope=None, word_embedding_dim=50, pos_embedding_dim=5, max_length=120):
    w_embedding = word_embedding(word, word_vec_mat, var_scope=var_scope, word_embedding_dim=word_embedding_dim)
    p_embedding = pos_embedding(pos1, pos2, var_scope=var_scope, pos_embedding_dim=pos_embedding_dim, max_length=max_length)
    return tf.concat([w_embedding, p_embedding], -1)
