import opennre
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework, debug_encoder
import argparse

word2id = json.load(open('pretrain/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('pretrain/glove/glove.6B.50d_mat.npy')
rel2id = json.load(open('benchmark/wiki_distant/wiki_distant_rel2id.json'))
sentence_encoder = opennre.encoder.PCNNEncoder(token2id=word2id,
                                             max_length=120,
                                             word_size=50,
                                             position_size=5,
                                             hidden_size=230,
                                             blank_padding=True,
                                             kernel_size=3,
                                             padding_size=1,
                                             word2vec=word2vec)
model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)
framework = opennre.framework.BagRE(
    train_path='benchmark/wiki_distant/wiki_distant_train.txt',
    val_path='benchmark/wiki_distant/wiki_distant_val.txt',
    test_path='benchmark/wiki_distant/wiki_distant_test.txt',
    model=model,
    ckpt='',
    batch_size=160,
    max_epoch=20,
    lr=0.5,
    weight_decay=0,
    opt='sgd')
print('Finish framework')
# Test
framework.load_state_dict(torch.load('./ckpt/wikidistant_pcnn_att.pth.tar.backup')['state_dict'])
print('Finish loading')
query = [
    {'text': 'Jennifer Gates, daughter of Bill Gates, was doing eleventh grade', 
        'h': {'name': 'Bill Gates', 'pos': [28, 38]},
        't': {'name': 'Jennifer Gates', 'pos': [0, 13]}}]
print(framework.infer(query))
