# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework

ckpt = 'ckpt/wiki80_bertentity_softmax.pth.tar'
rel2id = json.load(open('benchmark/wiki80/wiki80_rel2id.json'))
sentence_encoder = opennre.encoder.BERTEntityEncoder(
    max_length=80, pretrain_path='pretrain/bert-base-uncased')
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
framework = opennre.framework.SentenceRE(
    train_path='benchmark/wiki80/wiki80_train.txt',
    val_path='benchmark/wiki80/wiki80_val.txt',
    test_path='benchmark/wiki80/wiki80_val.txt',
    model=model,
    ckpt=ckpt,
    batch_size=64,
    max_epoch=10,
    lr=2e-5,
    opt='bert_adam')
# Train
framework.train_model()
# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print('Accuracy on test set: {}'.format(result['acc']))
