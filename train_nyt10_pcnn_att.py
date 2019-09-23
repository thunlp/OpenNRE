# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework, debug_encoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.5, type=float, help='learning rate') 
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--max_epoch', default=20, type=int, help='max number of training epoches')
parser.add_argument('--hidden_size', default=230, type=int, help='dim of the hidden states')
parser.add_argument('--max_length', default=120, type=int, help='max length of sentences')
parser.add_argument('--batch_size', default=160, type=int, help='batch size')
parser.add_argument('--opt', default='sgd', help='type of the optimizer')
parser.add_argument('--only_test', action='store_true', help='only test')
parser.add_argument('--ckpt', default='ckpt/nyt10_pcnn_att_new2.pth.tar')
args = parser.parse_args()

word2id = json.load(open('pretrain/glove/glove.6B.50d_word2id.json'))
word2vec = np.load('pretrain/glove/glove.6B.50d_mat.npy')
rel2id = json.load(open('benchmark/nyt10/nyt10_rel2id.json'))
# sentence_encoder = debug_encoder.PCNNEncoder(len(word2id), word2id, 120, word2vec=word2vec, hidden_size=230)
sentence_encoder = opennre.encoder.PCNNEncoder(token2id=word2id,
                                             max_length=args.max_length,
                                             word_size=50,
                                             position_size=5,
                                             hidden_size=args.hidden_size,
                                             blank_padding=True,
                                             kernel_size=3,
                                             padding_size=1,
                                             word2vec=word2vec)
model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)
framework = opennre.framework.BagRE(
    train_path='benchmark/nyt10/nyt10_train.txt',
    val_path='benchmark/nyt10/nyt10_val.txt',
    test_path='benchmark/nyt10/nyt10_test.txt',
    model=model,
    ckpt=args.ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt=args.opt)
# Train
if not args.only_test:
    framework.train_model()
# Test
framework.load_state_dict(torch.load(args.ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print('AUC on test set: {}'.format(result['auc']))
