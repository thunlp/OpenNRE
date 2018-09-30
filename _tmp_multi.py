import nrekit
import numpy as np
import tensorflow as tf

# train_loader = nrekit.data_loader.json_file_data_loader('data/train.json', './data/nyt_word_vec.json', './data/rel2id.json', mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG, shuffle=True, reprocess=False)
# test_loader = nrekit.data_loader.json_file_data_loader('data/test.json', './data/nyt_word_vec.json', './data/rel2id.json', mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG, shuffle=False, reprocess=False)

train_loader = nrekit.data_loader.json_file_data_loader('data_lyk_json/train_nyt_lyk.json', './data_lyk_json/nyt_word_vec.json', './data_lyk_json/rel2id.json', mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG, shuffle=True, reprocess=False)
test_loader = nrekit.data_loader.json_file_data_loader('data_lyk_json/test_nyt_lyk.json', './data_lyk_json/nyt_word_vec.json', './data_lyk_json/rel2id.json', mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG, shuffle=False, reprocess=False)
# test_loader = nrekit.data_loader.npy_data_loader('./data_old', 'test', mode=nrekit.data_loader.npy_data_loader.MODE_ENTPAIR_BAG, shuffle=False)

# train_loader = nrekit.data_loader.npy_data_loader('./data_old', 'train', mode=nrekit.data_loader.npy_data_loader.MODE_RELFACT_BAG, shuffle=True)
# test_loader = nrekit.data_loader.npy_data_loader('./data_old', 'test', mode=nrekit.data_loader.npy_data_loader.MODE_ENTPAIR_BAG, shuffle=False)

f = nrekit.framework.re_framework(train_loader, test_loader)

class pcnn_att(nrekit.framework.re_model):
    def __init__(self, train_data_loader, batch_size, max_length=120):
        nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")

        x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)
        x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
        x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
        # self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
        # self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
        # self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot, keep_prob=0.5)
        # self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot, keep_prob=1.0)
        self._train_logit, train_repre = nrekit.network.selector.bag_maximum(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
        self._test_logit, test_repre = nrekit.network.selector.bag_maximum(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)

        self._test_logit = tf.nn.softmax(self._test_logit)
        self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot, weights_table=self.get_weights())

    def loss(self):
        return self._loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
            for i in range(len(self.train_data_loader.data_rel)):
                _weights_table[self.train_data_loader.data_rel[i]] += 1.0 
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
            print("Finish calculating")
        return weights_table

f .train(pcnn_att, ckpt_dir='tmp_ckpt', model_name='pcnn_max', max_epoch=60, gpu_nums=1)
#f.train(model,ckpt_dir='tmp_ckpt', model_name='pcnn_att', max_epoch=40)

