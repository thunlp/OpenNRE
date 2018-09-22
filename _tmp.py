import nrekit
import numpy as np
import tensorflow as tf

train_loader = nrekit.data_loader.json_file_data_loader('data/train.json', './data/nyt_data.json', mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG, shuffle=True, reprocess=False)
test_loader = nrekit.data_loader.json_file_data_loader('data/test.json', './data/nyt_data.json', mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG, shuffle=False, reprocess=False)

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

def train(f):
    x = nrekit.network.embedding.word_position_embedding(f.word, f.word_vec_mat, f.pos1, f.pos2)
    x = nrekit.network.encoder.pcnn(x, f.pos1, f.pos2)
    logit, repre = nrekit.network.selector.bag_maximum(x, f.scope, f.label, f.rel_tot, True)
    loss = nrekit.network.classifier.softmax_cross_entropy(logit, f.label, f.rel_tot, weights_table=get_weights(f))
    output = nrekit.network.classifier.output(logit)
    return loss, output

def test(framework):
    x = nrekit.network.embedding.word_position_embedding(f.word, f.word_vec_mat, f.pos1, f.pos2)
    x = nrekit.network.encoder.pcnn(x, f.pos1, f.pos2)
    logit, repre = nrekit.network.selector.bag_attention(x, f.scope, f.label, f.rel_tot, False)
    output = nrekit.network.classifier.output(logit)
    return output

train_loss, train_output = train(f)
test_output = test(f)
f.train(train_loss, train_output, test_output, ckpt_dir='tmp_ckpt', model_name='pcnn_ave_test', max_epoch=5)
