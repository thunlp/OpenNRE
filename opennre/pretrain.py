from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np

root_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/"
default_root_path = os.path.join(os.getenv('HOME'), '.opennre')

def check_root(root_path=default_root_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, 'benchmark'))
        os.mkdir(os.path.join(root_path, 'pretrain'))
        os.mkdir(os.path.join(root_path, 'pretrain/nre'))

def download_wiki80(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/wiki80')):
        os.mkdir(os.path.join(root_path, 'benchmark/wiki80'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' ' + root_url + 'opennre/benchmark/wiki80/wiki80_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' ' + root_url + 'opennre/benchmark/wiki80/wiki80_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' ' + root_url + 'opennre/benchmark/wiki80/wiki80_val.txt')

def download_nyt10(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/nyt10')):
        os.mkdir(os.path.join(root_path, 'benchmark/nyt10'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' ' + root_url + 'opennre/benchmark/nyt10/nyt10_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' ' + root_url + 'opennre/benchmark/nyt10/nyt10_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' ' + root_url + 'opennre/benchmark/nyt10/nyt10_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' ' + root_url + 'opennre/benchmark/nyt10/nyt10_val.txt')

def download_semeval(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/semeval')):
        os.mkdir(os.path.join(root_path, 'benchmark/semeval'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/semeval') + ' ' + root_url + 'opennre/benchmark/semeval/semeval_val.txt')

def download_glove(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/glove')):
        os.mkdir(os.path.join(root_path, 'pretrain/glove'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'opennre/pretrain/glove/glove.6B.50d_mat.npy')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' ' + root_url + 'opennre/pretrain/glove/glove.6B.50d_word2id.json')

def download_bert_base_uncased(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/bert-base-uncased')):
        os.mkdir(os.path.join(root_path, 'pretrain/bert-base-uncased'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/config.json')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/pytorch_model.bin')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' ' + root_url + 'opennre/pretrain/bert-base-uncased/vocab.txt')

def download_pretrain(model_name, root_path=default_root_path):
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    if not os.path.exists(ckpt):
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/nre')  + ' ' + root_url + 'opennre/pretrain/nre/' + model_name + '.pth.tar')

def download(name, root_path=default_root_path):
    if not os.path.exists(os.path.join(root_path, 'benchmark')):
        os.mkdir(os.path.join(root_path, 'benchmark'))
    if not os.path.exists(os.path.join(root_path, 'pretrain')):
        os.mkdir(os.path.join(root_path, 'pretrain'))
    if name == 'nyt10':
        download_nyt10(root_path=root_path)
    elif name == 'semeval':
        download_semeval(root_path=root_path)
    elif name == 'wiki80':
        download_wiki80(root_path=root_path)
    elif name == 'glove':
        download_glove(root_path=root_path)
    elif name == 'bert_base_uncased':
        download_bert_base_uncased(root_path=root_path)
    else:
        raise Exception('Cannot find corresponding data.')

def get_model(model_name, root_path=default_root_path):
    check_root()
    ckpt = os.path.join(root_path, 'pretrain/nre/' + model_name + '.pth.tar')
    if model_name == 'wiki80_cnn_softmax':
        download_pretrain(model_name, root_path=root_path)
        download('glove', root_path=root_path)
        download('wiki80', root_path=root_path)
        wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        sentence_encoder = encoder.CNNEncoder(token2id=wordi2d,
                                                     max_length=40,
                                                     word_size=50,
                                                     position_size=5,
                                                     hidden_size=230,
                                                     blank_padding=True,
                                                     kernel_size=3,
                                                     padding_size=1,
                                                     word2vec=word2vec,
                                                     dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    elif model_name == 'wiki80_bert_softmax':
        download_pretrain(model_name, root_path=root_path)
        download('bert_base_uncased', root_path=root_path)
        download('wiki80', root_path=root_path)
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        sentence_encoder = encoder.BERTEncoder(
            max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
        return m
    else:
        raise NotImplementedError
