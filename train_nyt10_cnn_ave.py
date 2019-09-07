# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework

ckpt = 'ckpt/nyt10_pcnn_att.pth.tar'
wordi2d = json.load(open('pretrain/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('pretrain/glove/glove.6B.50d_mat.npy')
rel2id = json.load(open('benchmark/nyt10/nyt10_rel2id.json'))
sentence_encoder = opennre.encoder.CNNEncoder(token2id=wordi2d,
                                             max_length=120,
                                             word_size=50,
                                             position_size=5,
                                             hidden_size=230,
                                             blank_padding=True,
                                             kernel_size=3,
                                             padding_size=1,
                                             word2vec=word2vec,
                                             dropout=0.0)
model = opennre.model.BagAverage(sentence_encoder, len(rel2id), rel2id)
framework = opennre.framework.BagRE(
    train_path='benchmark/nyt10/nyt10_train.txt',
    val_path='benchmark/nyt10/nyt10_test.txt',
    test_path='benchmark/nyt10/nyt10_test.txt',
    model=model,
    ckpt=ckpt,
    batch_size=160,
    max_epoch=60,
    lr=0.5,
    weight_decay=0,
    opt='sgd')
# Train
framework.train_model()
# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print('AUC on test set: {}'.format(result['auc']))
