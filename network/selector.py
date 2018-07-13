import tensorflow as tf
import numpy as np

class Selector(object):

    def __init__(self, num_classes, is_training = False, drop_prob = None):
        self.num_classes = num_classes
        self.is_training = is_training
        self.dropout = drop_prob

    def __call__(self, is_training = False, drop_prob = None):
        self.is_training = is_training
        self.dropout = drop_prob

    def __dropout__(self, x):
        if self.dropout:
            return tf.layers.dropout(x, rate = self.dropout, training = self.is_training)
        else:
            return x

    def __logits__(self, x, var_scope = None, reuse = None):
        with tf.variable_scope(var_scope or 'logits', reuse = reuse):
            relation_matrix = tf.get_variable('relation_matrix', [self.num_classes, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [self.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(x, tf.transpose(relation_matrix)) + bias
        return logits

    def __attention_train_logits__(self, x, query, var_scope = None, reuse = None):
        with tf.variable_scope(var_scope or 'attention_logits', reuse = reuse):
            relation_matrix = tf.get_variable('relation_matrix', [self.num_classes, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [self.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            current_attention = tf.nn.embedding_lookup(relation_matrix, query)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
        return attention_logit

    def __attention_test_logits__(self, x, var_scope = None, reuse = None):
        with tf.variable_scope(var_scope or 'attention_logits', reuse = reuse):
            relation_matrix = tf.get_variable('relation_matrix', [self.num_classes, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [self.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        return tf.matmul(x, tf.transpose(relation_matrix))

    def no_bag(self, x):
        with tf.name_scope("no_bag"):
            x = self.__dropout__(x)
        return __logits__(x, "no_bag_logits", False), x

    def attention(self, x, scope, query, dropout_before = False):
        with tf.name_scope("attention"):
            if self.is_training:
                if dropout_before:
                    x = self.__dropout__(x)
                attention_logit = self.__attention_train_logits__(x, query, "attention_logits", False)
                tower_repre = []
                for i in range(scope.shape[0] - 1):
                    sen_matrix = x[scope[i]:scope[i + 1]]
                    attention_score = tf.nn.softmax(tf.reshape(attention_logit[scope[i]:scope[i + 1]], [1, -1]))
                    final_repre = tf.squeeze(tf.matmul(attention_score, sen_matrix))
                    tower_repre.append(final_repre)
                if not dropout_before:
                    stack_repre = self.__dropout__(tf.stack(tower_repre))
                else:
                    stack_repre = tf.stack(tower_repre)
                return self.__logits__(stack_repre, "attention_logits", True), stack_repre
            else:
                test_attention_logit = self.__attention_test_logits__(x, "attention_logits", False)
                test_tower_output = []
                test_repre = []
                for i in range(scope.shape[0] - 1):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[scope[i]:scope[i+1],:]))
                    final_repre = tf.matmul(test_attention_score, x[scope[i]:scope[i+1]])
                    logits = self.__logits__(final_repre, "attention_logits", True)
                    test_repre.append(final_repre)
                    test_tower_output.append(tf.diag_part(tf.nn.softmax(logits)))
                test_repre = tf.reshape(tf.stack(test_repre), [scope.shape[0] - 1, self.num_classes, -1])
                test_output = tf.reshape(tf.stack(test_tower_output), [scope.shape[0] - 1, self.num_classes])
                return test_output, test_repre

    def average(self, x, scope, dropout_before = False):
        with tf.name_scope("average"):
            if dropout_before:
                x = self.__dropout__(x)
            tower_repre = []
            for i in range(scope.shape[0] - 1):
                repre_mat = x[scope[i]:scope[i + 1]]
                repre = tf.reduce_mean(repre_mat, axis=0)
                tower_repre.append(repre)
            if not dropout_before:
                stack_repre = self.__dropout__(tf.stack(tower_repre))
            else:
                stack_repre = tf.stack(tower_repre)
        return __logits__(stack_repre, "average_logits", False), stack_repre

    def maximum(self, x, scope, dropout_before = False):
        with tf.name_scope("maximum"):
            if dropout_before:
                x = self.__dropout__(x)
            tower_repre = []
            for i in range(scope.shape[0] - 1):
                repre_mat = x[scope[i]:scope[i + 1]]
                logits = __logits__(repre_mat, "maximum_logits", False)
                j = tf.argmax(tf.reduce_max(logits, axis = 1))
                tower_repre.append(repre_mat[j])
            if not dropout_before:
                stack_repre = self.__dropout__(tf.stack(tower_repre))
            else:
                stack_repre = tf.stack(tower_repre)
        return __logits__(stack_repre, "maximum_logits", True), stack_repre
