# coding:utf-8
import sys, json
import torch
import os
import numpy as np
import opennre
import argparse
import logging
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--result', default='', 
        help='Result name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'], 
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true', 
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'wiki_distant', 'nyt10'],
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Bag related
parser.add_argument('--bag_size', type=int, default=4,
        help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=16, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')

# Exp
parser.add_argument('--aggr', default='att', choices=['one', 'att', 'avg'])
parser.add_argument('--use_diag', action='store_true', help='Use diag embedding for ATT')


# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)
args = parser.parse_args()

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

if args.dataset != 'none':
    opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
    args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
    if not os.path.exists(args.val_file):
        logging.info("Cannot find the validation file. Use the test file instead.")
        args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
    args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))


# Define the sentence encoder
if args.pooler == 'entity':
    sentence_encoder = opennre.encoder.BERTEntityEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
elif args.pooler == 'cls':
    sentence_encoder = opennre.encoder.BERTEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
else:
    raise NotImplementedError


# Define the model
if args.aggr == 'att':
    model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id, use_diag=args.use_diag)
elif args.aggr == 'avg':
    model = opennre.model.BagAverage(sentence_encoder, len(rel2id), rel2id)
elif args.aggr == 'one':
    model = opennre.model.BagOne(sentence_encoder, len(rel2id), rel2id)
else:
    raise NotImplementedError

# Define the whole training framework
framework = opennre.framework.BagRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt="adamw",
    bag_size=args.bag_size
)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
logging.info('Test set results:')
logging.info('AUC: %.5f' % (result['auc']))
logging.info('Maximum micro F1: %.5f' % (result['max_micro_f1']))
logging.info('Maximum macro F1: %.5f' % (result['max_macro_f1']))
logging.info('Micro F1: %.5f' % (result['micro_f1']))
logging.info('Macro F1: %.5f' % (result['macro_f1']))
logging.info('P@100: %.5f' % (result['p@100']))
logging.info('P@200: %.5f' % (result['p@200']))
logging.info('P@300: %.5f' % (result['p@300']))


# Save precision/recall points
np.save('result/{}_p.npy'.format(args.result), result['np_prec'])
np.save('result/{}_r.npy'.format(args.result), result['np_rec'])
json.dump(result['max_micro_f1_each_relation'], open('result/{}_mmicrof1_rel.json'.format(args.result), 'w'), ensure_ascii=False)
