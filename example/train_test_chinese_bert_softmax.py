import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mask_entity', action='store_true', help='Mask entity mentions')
args = parser.parse_args()

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/test_chinese_bert_softmax.pth.tar'

# Check data
rel2id = json.load(open(os.path.join(root_path, 'benchmark/test_chinese/test_chinese_rel2id.json')))

# Define the sentence encoder
sentence_encoder = opennre.encoder.BERTEncoder(
    max_length=80, 
    pretrain_path=os.path.join(root_path, 'pretrain/chinese_wwm_pytorch'),
    mask_entity=args.mask_entity
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=os.path.join(root_path, 'benchmark/test_chinese/test_chinese_train.txt'),
    val_path=os.path.join(root_path, 'benchmark/test_chinese/test_chinese_val.txt'),
    test_path=os.path.join(root_path, 'benchmark/test_chinese/test_chinese_val.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=64, # Modify the batch size w.r.t. your device
    max_epoch=7,
    lr=2e-5,
    opt='adamw'
)

# Train the model
framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
