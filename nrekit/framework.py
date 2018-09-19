import tensorflow as tf

class re_framework:
    def __init__(self, data_loader, max_length=120):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='scope')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        self.word_vec_mat = data_loader.word_vec_mat

    def one_step(self, batch_data, sess_array, keep_prob=1.0):
        feed_dict = {
            self.word: batch_data['word'],
            self.pos1: batch_data['pos1'],
            self.pos2: batch_data['pos2'],
            self.label: batch_data['label'],
            self.scope: batch_data['scope'],
            self.length: batch_data['length'],
            self.keep_prob: keep_prob
        }
        with tf.Session() as sess:
            result = sess.run(sess_array, feed_dict)
        return result

    
