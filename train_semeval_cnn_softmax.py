#coding:utf-8
import torch
import numpy as np
import json
import nrekit
from nrekit import encoder, model, framework

wordi2d = json.load(open('benchmark/word2id_glove50.json'))
word2vec = np.load('benchmark/word2vec_glove50.npy')
rel2id = json.load(open('benchmark/semeval/semeval_rel2id.json'))
sentence_encoder = nrekit.encoder.CNNEncoder(token2id = wordi2d, 
											 max_length = 50, 
											 word_size = 400,
											 position_size = 70, 
											 hidden_size = 1000,
											 blank_padding = True,
											 kernel_size = 3, 
											 padding_size = 1,
											 dropout = 0.5)
model = nrekit.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
framework = nrekit.framework.SentenceRE(train_path = 'benchmark/semeval/semeval_train.txt', 
										val_path = 'benchmark/semeval/semeval_test.txt', 
										model = model,
										ckpt = 'ckpt/wiki80_cnn.pth.tar',
										batch_size = 128, 
										max_epoch = 20, 
										lr = 0.5, 
										weight_decay = 1e-5, 
										opt='sgd')
framework.train_model()