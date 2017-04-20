import tensorflow as tf
import numpy as np


class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.num_steps = 70
        self.num_epochs = 3
        self.num_classes = 53
        self.gru_size = 230
        self.keep_prob = 0.5
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        # the number of entity pairs of each batch during training or testing
        self.big_num = 50


class GRU:
    def __init__(self, is_training, word_embeddings, settings):

        self.num_steps = num_steps = settings.num_steps
        self.vocab_size = vocab_size = settings.vocab_size
        self.num_classes = num_classes = settings.num_classes
        self.gru_size = gru_size = settings.gru_size
        self.big_num = big_num = settings.big_num

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[big_num + 1], name='total_shape')
        total_num = self.total_shape[-1]

        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [settings.pos_num, settings.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [settings.pos_num, settings.pos_size])

        attention_w = tf.get_variable('attention_omega', [gru_size, 1])
        sen_a = tf.get_variable('attention_A', [gru_size])
        sen_r = tf.get_variable('query_r', [gru_size, 1])
        relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, gru_size])
        sen_d = tf.get_variable('bias_d', [self.num_classes])

        gru_cell_forward = tf.contrib.rnn.GRUCell(gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(gru_size)

        if is_training and settings.keep_prob < 1:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=settings.keep_prob)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=settings.keep_prob)

        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * settings.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * settings.num_layers)

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0

        self._initial_state_forward = cell_forward.zero_state(total_num, tf.float32)
        self._initial_state_backward = cell_backward.zero_state(total_num, tf.float32)

        # embedding layer
        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, self.input_word),
                                                   tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
                                                   tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)])
        inputs_backward = tf.concat(axis=2,
                                    values=[tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [1])),
                                            tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [1])),
                                            tf.nn.embedding_lookup(pos1_embedding,
                                                                   tf.reverse(self.input_pos2, [1]))])

        outputs_forward = []

        state_forward = self._initial_state_forward

        # Bi-GRU layer
        with tf.variable_scope('GRU_FORWARD'):
            for step in range(num_steps):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(step > 0)):
                    (cell_output_forward, state_forward) = cell_forward(inputs_forward[:, step, :], state_forward)
                    outputs_forward.append(cell_output_forward)

        outputs_backward = []

        state_backward = self._initial_state_backward
        with tf.variable_scope('GRU_BACKWARD'):
            for step in range(num_steps):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(step > 0)):
                    (cell_output_backward, state_backward) = cell_backward(inputs_backward[:, step, :], state_backward)
                    outputs_backward.append(cell_output_backward)

        output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward), [total_num, num_steps, gru_size])
        output_backward = tf.reverse(
            tf.reshape(tf.concat(axis=1, values=outputs_backward), [total_num, num_steps, gru_size]),
            [1])

        # word-level attention layer
        output_h = tf.add(output_forward, output_backward)
        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * num_steps, gru_size]), attention_w),
                       [total_num, num_steps])), [total_num, 1, num_steps]), output_h), [total_num, gru_size])

        # sentence-level attention layer
        for i in range(big_num):

            sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
            batch_size = self.total_shape[i + 1] - self.total_shape[i]

            sen_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
                           [1, batch_size]))

            sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [gru_size, 1]))
            sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [self.num_classes]), sen_d))

            self.prob.append(tf.nn.softmax(sen_out[i]))

            with tf.name_scope("output"):
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):
                self.loss.append(
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.input_y[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            # tf.summary.scalar('loss',self.total_loss)
            # tf.scalar_summary(['loss'],[self.total_loss])
            with tf.name_scope("accuracy"):
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                                   name="accuracy"))

        # tf.summary.scalar('loss',self.total_loss)
        tf.summary.scalar('loss', self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)
