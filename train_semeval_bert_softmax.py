# coding:utf-8
import torch
import numpy as np
import json
import nrekit
from nrekit import encoder, model, framework

ckpt = 'ckpt/semeval_bert_softmax.pth.tar'
rel2id = json.load(open('benchmark/semeval/semeval_rel2id.json'))
sentence_encoder = nrekit.encoder.BERTEncoder(
    max_length=80, pretrain_path='pretrain/bert-base-uncased')
model = nrekit.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
framework = nrekit.framework.SentenceRE(
    train_path='benchmark/semeval/semeval_train.txt',
    val_path='benchmark/semeval/semeval_val.txt',
    test_path='benchmark/semeval/semeval_test.txt',
    model=model,
    ckpt=ckpt,
    batch_size=64,
    max_epoch=10,
    lr=3e-5,
    opt='bert_adam')
# Train
framework.train_model()
# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print('Accuracy on test set: {}'.format(result['acc']))
