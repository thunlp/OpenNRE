# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mask_entity', action='store_true', help='Mask entity mentions')
parser.add_argument('--pretrain_path', default='bert-base-uncased', help='Pre-trained ckpt path (hugginface)')
parser.add_argument('--ckpt', default='tacred_bert_softmax', help='Checkpoint name')
args = parser.parse_args()

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

opennre.download('bert_base_uncased', root_path=root_path)
rel2id = json.load(open(os.path.join(root_path, 'benchmark/tacred/tacred_rel2id.json')))

# Define the sentence encoder
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=128, 
    pretrain_path=args.pretrain_path,
    mask_entity=args.mask_entity
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=os.path.join(root_path, 'benchmark/tacred/tacred_train.txt'),
    val_path=os.path.join(root_path, 'benchmark/tacred/tacred_val.txt'),
    test_path=os.path.join(root_path, 'benchmark/tacred/tacred_test.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=64, # Modify the batch size w.r.t. your device
    max_epoch=10,
    lr=2e-5,
    opt='adamw'
)

# Train the model
framework.train_model('micro_f1')

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
print('Micro Precision: {}'.format(result['micro_p']))
print('Micro Recall: {}'.format(result['micro_r']))
print('Micro F1: {}'.format(result['micro_f1']))
