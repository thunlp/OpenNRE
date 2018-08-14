import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys

tf.app.flags.DEFINE_string('export_path','./data','path to data')

config_file = open(os.path.join('data', "config"), 'r')
config = json.loads(config_file.read())
config_file.close()

tf.app.flags.DEFINE_integer('max_length', config['fixlen'], 'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1, 'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', len(config['relation2id']),'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')
tf.app.flags.DEFINE_integer('word_size', 50, 'word embedding size')

tf.app.flags.DEFINE_integer('batch_size',160,'entity numbers used each training time')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'path to store checkpoint')
tf.app.flags.DEFINE_string('test_result_dir', './test_result', 'path to store the test results')

tf.app.flags.DEFINE_string('model_name', 'pcnn_att', 'model\'s name')
tf.app.flags.DEFINE_string('epoch_range', '(5, 60)', 'checkpoint epoch range')

tf.app.flags.DEFINE_float('drop_prob', 0.5, 'dropout rate')
tf.app.flags.DEFINE_boolean('discard_only_one', False, 'use discard only one entity mode')

FLAGS = tf.app.flags.FLAGS

from framework import Framework 
def main(_):
    from model.pcnn_att import pcnn_att
    from model.cnn_att import cnn_att
    from model.pcnn_att_adv import pcnn_att_adv
    from model.pcnn_att_soft_label import pcnn_att_soft_label
    from model.pcnn_ave import pcnn_ave
    from model.pcnn_max import pcnn_max
    from model.pcnn import pcnn
    from model.cnn_ave import cnn_ave
    from model.cnn_max import cnn_max
    from model.cnn import cnn
    from model.rnn_att import rnn_att
    from model.rnn_max import rnn_max
    from model.rnn_ave import rnn_ave
    from model.rnn import rnn
    from model.birnn import birnn
    from model.birnn_max import birnn_max
    from model.birnn_ave import birnn_ave
    from model.birnn_att import birnn_att

    from model.pcnn_att_tanh import pcnn_att_tanh

    from model.pcnn_ave_adv import pcnn_ave_adv
    from model.pcnn_max_adv import pcnn_max_adv
    from model.cnn_ave_adv import cnn_ave_adv 
    from model.cnn_max_adv import cnn_max_adv
    from model.cnn_att_adv import cnn_att_adv
    from model.rnn_att_adv import rnn_att_adv
    from model.rnn_max_adv import rnn_max_adv
    from model.rnn_ave_adv import rnn_ave_adv
    from model.birnn_max_adv import birnn_max_adv  
    from model.birnn_ave_adv import birnn_ave_adv
    from model.birnn_att_adv import birnn_att_adv
    from model.pcnn_att_adam import pcnn_att_adam

    if not FLAGS.model_name in locals():
        exit()
    model = locals()[FLAGS.model_name]
    model(is_training=False)

if __name__ == "__main__":
    tf.app.run() 
