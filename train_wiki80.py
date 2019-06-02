import nrekit
import torch
import numpy as np
import json

wordi2d = json.load(open('data/word2id.json'))
word2vec = np.load('data/word2vec.npy')
rel2id = json.load(open('data/wiki80_rel2id.json'))
ckpt = torch.load('data/cnn_encoder.pth.tar')['state_dict']
sentence_encoder = nrekit.encoder.CNNEncoder(len(wordi2d), wordi2d, 40, word2vec=word2vec)
model = nrekit.models.CNNSoftmax(sentence_encoder, len(rel2id), rel2id)
model.train_model('data/wiki80_train.txt', 'data/wiki80_val.txt', 'ckpt/wiki80_cnn.pth.tar')