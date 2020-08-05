import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bag_size', type=int, default=0)
parser.add_argument('--ckpt', type=str, default='nyt10_pcnn_att')
parser.add_argument('--only_test', action='store_true',
        help='Only test')

# Hyper-parameters
parser.add_argument('--batch_size', default=160, type=int,
        help='Batch size')
parser.add_argument('--lr', default=0.1, type=float,
        help='Learning rate')
parser.add_argument('--optim', default='sgd', type=str,
        help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--max_epoch', default=100, type=int,
        help='Max number of training epochs')

args = parser.parse_args()

# Some basic settings
root_path = '.'
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

# Check data
opennre.download('nyt10', root_path=root_path)
opennre.download('glove', root_path=root_path)
rel2id = json.load(open(os.path.join(root_path, 'benchmark/nyt10/nyt10_rel2id.json')))
wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.PCNNEncoder(
    token2id=wordi2d,
    max_length=120,
    word_size=50,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.5
)

# Define the model
model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.BagRE(
    train_path='benchmark/nyt10/nyt10_train.txt',
    val_path='benchmark/nyt10/nyt10_val.txt',
    test_path='benchmark/nyt10/nyt10_test.txt',
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt=args.optim,
    bag_size=args.bag_size)

# Train the model
if not args.only_test:
    framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('AUC on test set: {}'.format(result['auc']))

