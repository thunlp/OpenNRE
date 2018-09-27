import tensorflow as tf
import numpy as np
import math

def __dropout__(x, keep_prob=1.0):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)

def __init_mask_embedding__(shape, dtype=tf.int32, partition_info=None):
    assert(len(shape) == 3 and shape[1] * shape[1] == shape[0] and shape[2] == 3) # (max_length ^ 2, max_length, 3)
    max_length = shape[1]
    mask_embedding = np.zeros(shape, dtype=np.int32) 
    for pos1 in range(max_length):
        for pos2 in range(max_length):
            idx = pos1 * max_length + pos2
            pos_first = min(pos1, pos2)
            pos_second = max(pos1, pos2)
            mask_embedding[idx, :pos_first, 0] = 1
            mask_embedding[idx, pos_first:pos_second, 1] = 1
            mask_embedding[idx, pos_second:, 2] = 1
    return mask_embedding

def __mask__(pos1, pos2):
    with tf.variable_scope("mask", reuse=tf.AUTO_REUSE):
        max_length = pos1.shape[-1]
        mask_embedding = tf.get_variable("mask_embedding", shape=[max_length * max_length, max_length, 3], initializer=__init_mask_embedding__, trainable=False)
        return tf.nn.embedding_lookup(mask_embedding, pos1[:, 0] * max_length + pos2[:, 0])

def __pooling__(x):
    return tf.reduce_max(x, axis=-2)

def __piecewise_pooling__(x, pos1, pos2):
    mask = __mask__(pos1, pos2)
    max_length = pos1.shape[-1]
    hidden_size = x.shape[-1]
    x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
    return tf.reshape(x, [-1, hidden_size * 3])

def __piecewise_pooling_with_mask__(x, mask):
    mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding, mask)
    max_length = pos1.shape[-1]
    hidden_size = x.shape[-1]
    x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
    return tf.reshape(x, [-1, hidden_size * 3])

def __cnn_cell__(x, hidden_size=230, kernel_size=3, stride_size=1):
    x = tf.layers.conv1d(inputs=x, 
                         filters=hidden_size, 
                         kernel_size=kernel_size, 
                         strides=stride_size, 
                         padding='same', 
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x

def cnn(x, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __pooling__(x)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def pcnn(x, pos1, pos2, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __piecewise_pooling__(x, pos1, pos2)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def pcnn_with_mask(x, mask, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __piecewise_pooling_with_mask__(x, mask)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def __rnn_cell__(hidden_size, cell_name='lstm'):
    if isinstance(cell_name, list) or isinstance(cell_name, tuple):
        if len(cell_name) == 1:
            return __rnn_cell__(hidden_size, cell_name[0])
        cells = [self.__rnn_cell__(hidden_size, c) for c in cell_name]
        return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    if cell_name.lower() == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    elif cell_name.lower() == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_size)
    raise NotImplemented

def rnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        x = __dropout__(x, keep_prob)
        cell = __rnn_cell__(hidden_size, cell_name)
        _, states = tf.nn.dynamic_rnn(cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        if isinstance(states, tuple):
            states = states[0]
        return states

def birnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "birnn", reuse=tf.AUTO_REUSE):
        x = __dropout__(x, keep_prob)
        fw_cell = __rnn_cell__(hidden_size, cell_name)
        bw_cell = __rnn_cell__(hidden_size, cell_name)
        _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-bi-rnn')
        fw_states, bw_states = states
        if isinstance(fw_states, tuple):
            fw_states = fw_states[0]
            bw_states = bw_states[0]
        return tf.concat([fw_states, bw_states], axis=1)

