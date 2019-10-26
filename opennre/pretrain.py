from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np

def get_model(model_name):
    if model_name == 'wiki80_cnn_softmax':
        ckpt = 'pretrain/nre/wiki80_cnn_softmax.pth.tar'
        if not os.path.exists(ckpt):
            os.system('wget -P pretrain/nre wiki80_cnn_softmax.pth.tar http://193.112.16.83:8080/opennre/pretrain/nre/wiki80_cnn_softmax.pth.tar')
        if not os.path.exists('pretrain/glove'):
            os.system('mkdir pretrain/glove')
            os.system('wget -P pretrain/glove http://193.112.16.83:8080/opennre/pretrain/glove/glove.6B.50d_mat.npy')
            os.system('wget -P pretrain/glove http://193.112.16.83:8080/opennre/pretrain/glove/glove.6B.50d_word2id.json')
        if not os.path.exists('benchmark/wiki80'):
            os.system('mkdir benchmark/wiki80')
            os.system('wget -P benchmark/wiki80 http://193.112.16.83:8080/opennre/benchmark/wiki80/wiki80_rel2id.json')
            os.system('wget -P benchmark/wiki80 http://193.112.16.83:8080/opennre/benchmark/wiki80/wiki80_train.txt')
            os.system('wget -P benchmark/wiki80 http://193.112.16.83:8080/opennre/benchmark/wiki80/wiki80_val.txt')
        wordi2d = json.load(open('pretrain/glove/glove.6B.50d_word2id.json'))
        word2vec = np.load('pretrain/glove/glove.6B.50d_mat.npy')
        rel2id = json.load(open('benchmark/wiki80/wiki80_rel2id.json'))
        sentence_encoder = encoder.CNNEncoder(token2id=wordi2d,
                                                     max_length=40,
                                                     word_size=50,
                                                     position_size=5,
                                                     hidden_size=230,
                                                     blank_padding=True,
                                                     kernel_size=3,
                                                     padding_size=1,
                                                     word2vec=word2vec,
                                                     dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt)['state_dict'])
        return m
    else:
        raise NotImplementedError
