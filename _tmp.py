import nrekit
import numpy as np
import tensorflow as tf

train_loader = nrekit.data_loader.json_file_data_loader('data/train.json', './data/nyt_data.json', './data/rel2id.json', mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG, shuffle=True, reprocess=False)
test_loader = nrekit.data_loader.json_file_data_loader('data/test.json', './data/nyt_data.json', './data/rel2id.json', mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG, shuffle=False, reprocess=False)

f = nrekit.framework.re_framework(train_loader, test_loader)

def get_weights(f):
    print("Calculating weights_table...")
    _weights_table = np.zeros((f.train_data_loader.rel_tot), dtype=np.float32)
    for i in range(len(f.train_data_loader.data_rel)):
        _weights_table[f.train_data_loader.data_rel[i]] += 1.0 
    _weights_table = 1 / (_weights_table ** 0.05)
    weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
    print("Finish calculating")
    return weights_table

def model(f):
    x = nrekit.network.embedding.word_position_embedding(f.word, f.word_vec_mat, f.pos1, f.pos2)
    x = nrekit.network.encoder.pcnn(x, f.pos1, f.pos2)
    train_logit, train_repre = nrekit.network.selector.bag_attention(x, f.scope, f.ins_label, f.rel_tot, True)
    test_logit, test_repre = nrekit.network.selector.bag_attention(x, f.scope, f.ins_label, f.rel_tot, False)
    loss = nrekit.network.classifier.softmax_cross_entropy(train_logit, f.label, f.rel_tot, weights_table=get_weights(f))
    return loss, train_logit, test_logit

loss, train_logit, test_logit = model(f)
f.train(loss, train_logit, test_logit, ckpt_dir='tmp_ckpt', model_name='pcnn_att', max_epoch=40)
