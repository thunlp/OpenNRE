import tensorflow as tf
import numpy as np

def __dropout__(x):
    keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)

def __logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.matmul(x, tf.transpose(relation_matrix)) + bias
    return logit

def __attention_train_logit__(x, query, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    current_relation = tf.nn.embedding_lookup(relation_matrix, query)
    attention_logit = tf.reduce_sum(current_relation * x, -1) # sum[(n', hidden_size) \dot (n', hidden_size)] = (n)
    return attention_logit

def __attention_test_logit__(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    attention_logit = tf.matmul(x, tf.transpose(relation_matrix)) # (n', hidden_size) x (hidden_size, rel_tot) = (n', rel_tot)
    return attention_logit

def instance(x, rel_tot, var_scope=None):
    x = __dropout__(x)
    x = __logit__(x, rel_tot)
    return x

def bag_attention(x, scope, query, rel_tot, is_training, var_scope=None, dropout_before=False):
    with tf.variable_scope(var_scope or "attention", reuse=tf.AUTO_REUSE):
        if is_training: # training
            if dropout_before:
                x = __dropout__(x)
            bag_repre = []
            attention_logit = __attention_train_logit__(x, query, rel_tot)
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(attention_logit[scope[i][0]:scope[i][1]], -1)
                bag_repre.append(tf.squeeze(tf.matmul(tf.expand_dims(attention_score, 0), bag_hidden_mat))) # (1, n') x (n', hidden_size) = (1, hidden_size) -> (hidden_size)
            bag_repre = tf.stack(bag_repre)
            if not dropout_before:
                bag_repre = __dropout__(bag_repre)
            return __logit__(bag_repre, rel_tot), bag_repre
        else: # testing
            attention_logit = __attention_test_logit__(x, rel_tot) # (n, rel_tot)
            bag_repre = [] 
            bag_logit = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                attention_score = tf.nn.softmax(tf.transpose(attention_logit[scope[i][0]:scope[i][1], :]), -1) # softmax of (rel_tot, n')
                bag_repre_for_each_rel = tf.matmul(attention_score, bag_hidden_mat) # (rel_tot, n') \dot (n', hidden_size) = (rel_tot, hidden_size)
                bag_logit_for_each_rel = __logit__(bag_repre_for_each_rel, rel_tot) # -> (rel_tot, rel_tot)
                bag_repre.append(bag_repre_for_each_rel)
                bag_logit.append(tf.diag_part(tf.nn.softmax(bag_logit_for_each_rel, -1))) # could be improved by sigmoid?
            bag_repre = tf.stack(bag_repre)
            bag_logit = tf.stack(bag_logit)
            return bag_logit, bag_repre

def bag_average(x, scope, rel_tot, var_scope=None, dropout_before=False):
    with tf.variable_scope(var_scope or "average", reuse=tf.AUTO_REUSE):
        if dropout_before:
            x = __dropout__(x)
        bag_repre = []
        for i in range(scope.shape[0]):
            bag_hidden_mat = x[scope[i][0]:scope[i][1]]
            bag_repre.append(tf.reduce_mean(bag_hidden_mat, 0)) # (n', hidden_size) -> (hidden_size)
        bag_repre = tf.stack(bag_repre)
        if not dropout_before:
            bag_repre = __dropout__(bag_repre)
    return __logit__(bag_repre, rel_tot), bag_repre

def bag_maximum(x, scope, query, rel_tot, is_training, var_scope=None, dropout_before=False): # could be improved?
    with tf.variable_scope(var_scope or "maximum", reuse=tf.AUTO_REUSE):
        if is_training: # training
            if dropout_before:
                x = __dropout__(x)
            bag_repre = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                instance_logit = tf.nn.softmax(__logit__(bag_hidden_mat, rel_tot), -1) # (n', hidden_size) -> (n', rel_tot)
                j = tf.argmax(instance_logit[:, query[i]], output_type=tf.int32)
                bag_repre.append(bag_hidden_mat[j])
            bag_repre = tf.stack(bag_repre)
            if not dropout_before:
                bag_repre = __dropout__(bag_repre)
            return __logit__(bag_repre, rel_tot), bag_repre
        else: # testing
            if dropout_before:
                x = __dropout__(x)
            bag_repre = []
            bag_logit = []
            for i in range(scope.shape[0]):
                bag_hidden_mat = x[scope[i][0]:scope[i][1]]
                instance_logit = tf.nn.softmax(__logit__(bag_hidden_mat, rel_tot), -1) # (n', hidden_size) -> (n', rel_tot)
                bag_logit.append(tf.reduce_max(instance_logit, 0))
                bag_repre.append(bag_hidden_mat[0]) # fake max repre
            bag_logit = tf.stack(bag_logit)
            bag_repre = tf.stack(bag_repre)
            return bag_logit, bag_repre
