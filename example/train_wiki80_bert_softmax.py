import sys, json
import torch
import numpy as np
import opennre
from opennre import encoder, model, framework
# Some basic settings
sys.path.append('.')
ckpt = 'ckpt/wiki80_bert_softmax.pth.tar'
rel2id = json.load(open('benchmark/wiki80/wiki80_rel2id.json'))
# Define the sentence encoder.
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=80, pretrain_path='pretrain/bert-base-uncased')
# Define the model.
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
# Define the whole training framework.
framework = opennre.framework.SentenceRE(
    train_path = 'benchmark/wiki80/wiki80_train.txt',
    val_path   = 'benchmark/wiki80/wiki80_val.txt',
    test_path  = 'benchmark/wiki80/wiki80_val.txt',
    model      = model,
    ckpt       = ckpt,
    batch_size = 64,
    max_epoch  = 10,
    lr         = 2e-5,
    opt        = 'adamw')
# Train the model
framework.train_model()
# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
