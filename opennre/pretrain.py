from . import encoder
from . import model
from . import framework
import torch
import os
import sys
import json
import numpy as np

default_root_path = os.path.join(os.getenv('openNRE'), '.')

def check_root(root_path=default_root_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, 'benchmark'))
        os.mkdir(os.path.join(root_path, 'pretrain'))
        os.mkdir(os.path.join(root_path, 'ckpt'))

def download_wiki80(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/wiki80')):
        os.mkdir(os.path.join(root_path, 'benchmark/wiki80'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' http://193.112.16.83:8080/opennre/benchmark/wiki80/wiki80_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' http://193.112.16.83:8080/opennre/benchmark/wiki80/wiki80_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/wiki80') + ' http://193.112.16.83:8080/opennre/benchmark/wiki80/wiki80_val.txt')

def download_nyt10(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'benchmark/nyt10')):
        os.mkdir(os.path.join(root_path, 'benchmark/nyt10'))
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' http://193.112.16.83:8080/opennre/benchmark/nyt10/nyt10_rel2id.json')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' http://193.112.16.83:8080/opennre/benchmark/nyt10/nyt10_train.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' http://193.112.16.83:8080/opennre/benchmark/nyt10/nyt10_test.txt')
        os.system('wget -P ' + os.path.join(root_path, 'benchmark/nyt10') + ' http://193.112.16.83:8080/opennre/benchmark/nyt10/nyt10_val.txt')

def download_glove(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/glove')):
        os.mkdir(os.path.join(root_path, 'pretrain/glove'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' http://193.112.16.83:8080/opennre/pretrain/glove/glove.6B.50d_mat.npy')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/glove') +  ' http://193.112.16.83:8080/opennre/pretrain/glove/glove.6B.50d_word2id.json')

def download_bert_base_uncased(root_path=default_root_path):
    check_root()
    if not os.path.exists(os.path.join(root_path, 'pretrain/bert-base-uncased')):
        os.mkdir(os.path.join(root_path, 'pretrain/bert-base-uncased'))
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' http://193.112.16.83:8080/opennre/pretrain/bert-base-uncased/config.json')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' http://193.112.16.83:8080/opennre/pretrain/bert-base-uncased/pytorch_model.bin')
        os.system('wget -P ' + os.path.join(root_path, 'pretrain/bert-base-uncased') + ' http://193.112.16.83:8080/opennre/pretrain/bert-base-uncased/vocab.txt')

def download_pretrain(model_name, root_path=default_root_path):
    ckpt = os.path.join(root_path, 'ckpt/' + model_name + '.pth.tar')
    if not os.path.exists(ckpt):
        print("*"*20)
        print("下载ckpt")
        os.system('wget -P ' + os.path.join(root_path, 'ckpt/')  + ' http://193.112.16.83:8080/opennre/ckpt/' + model_name + '.pth.tar')

def get_model(model_name, root_path=default_root_path):
    check_root()
    ckpt = os.path.join(root_path, 'ckpt/' + model_name + '.pth.tar')
    
    if model_name == 'wiki80_cnn_softmax':
        print("*"*20+"taorui")
        download_pretrain(model_name)
        download_glove()
        download_wiki80()
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
        m.load_state_dict(torch.load(ckpt)['state_dict'])
        return m
    elif model_name == 'wiki80_bert_softmax':
        download_pretrain(model_name)
        download_bert_base_uncased()
        download_wiki80()
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
        sentence_encoder = encoder.BERTEncoder(
            max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/bert-base-uncased'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt)['state_dict'])
        return m
    elif model_name == 'test_chinese_bert_softmax':
        download_pretrain(model_name)
        download_bert_base_uncased()
        download_wiki80()
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/test_chinese/test_chinese_rel2id.json')))
        sentence_encoder = encoder.BERTEncoder(
            max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/chinese_wwm_pytorch'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt)['state_dict'])
        return m

    elif model_name == 'people_chinese_bert_softmax':
        download_pretrain(model_name)
        download_bert_base_uncased()
        download_wiki80()
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/people-relation/people-relation_rel2id.json')))
        sentence_encoder = encoder.BERTEncoder(
            max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/chinese_wwm_pytorch'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt)['state_dict'])
        return m 

    elif model_name == 'people_delunknown_chinese_bert_softmax':
        download_pretrain(model_name)
        download_bert_base_uncased()
        download_wiki80()
        rel2id = json.load(open(os.path.join(root_path, 'benchmark/people-relation-delunknow/people-relation_rel2id.json')))
        sentence_encoder = encoder.BERTEncoder(
            max_length=80, pretrain_path=os.path.join(root_path, 'pretrain/chinese_wwm_pytorch'))
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
        m.load_state_dict(torch.load(ckpt)['state_dict'])
        return m
    
    else:
        raise NotImplementedError
