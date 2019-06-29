from . import encoder
from . import framework
from . import models
from . import data_loader

__all__ = ['encoder', 'framework', 'models']

import json
import torch

def sentence_re_model(name='cnn_softmax_wiki80', cuda=False):
    if name == 'cnn_softmax_wiki80':
        wordi2d = json.load(open('pretrain/cnn_softmax_wiki80/word2id.json'))
        rel2id = json.load(open('pretrain/cnn_softmax_wiki80/wiki80_rel2id.json'))
        if cuda:
            ckpt = torch.load('pretrain/cnn_softmax_wiki80/cnn_softmax_wiki80.pth.tar')['state_dict']
        else:
            ckpt = torch.load('pretrain/cnn_softmax_wiki80/cnn_softmax_wiki80.pth.tar', map_location='cpu')['state_dict']
        sentence_encoder = encoder.CNNEncoder(len(wordi2d), wordi2d, 40)
        model = models.CNNSoftmax(sentence_encoder, len(rel2id), rel2id)
        model.load_state_dict(ckpt)
        model.eval()
        return model
    elif name == 'bert_softmax_wiki80':
        rel2id = json.load(open('pretrain/bert_softmax_wiki80/wiki80_rel2id.json'))
        if cuda:
            ckpt = torch.load('pretrain/bert_softmax_wiki80/bert_softmax_wiki80.pth.tar')['state_dict']
        else:
            ckpt = torch.load('pretrain/bert_softmax_wiki80/bert_softmax_wiki80.pth.tar', map_location='cpu')['state_dict']
        sentence_encoder = encoder.BERTEncoder(64, 'pretrain/bert-base-uncased')
        model = models.BERTSoftmax(sentence_encoder, len(rel2id), rel2id)
        model.load_state_dict(ckpt)
        model.eval()
        return model
    else:
        raise NotImplementedError

def fewshot_re_model(name='proto_cnn', cuda=False):
    if name == 'proto_cnn':
        wordi2d = json.load(open('pretrain/proto_cnn/word2id.json'))
        if cuda:
            ckpt = torch.load('pretrain/proto_cnn/proto_cnn.pth.tar')['state_dict']
        else:
            ckpt = torch.load('pretrain/proto_cnn/proto_cnn.pth.tar', map_location='cpu')['state_dict']
        sentence_encoder = encoder.CNNEncoder(len(wordi2d), wordi2d, 40)
        model = models.ProtoNetwork(sentence_encoder)
        model.load_state_dict(ckpt)
        model.eval()
        return model
    elif name == 'bert_pair':
        if cuda:
            ckpt = torch.load('pretrain/bert_pair/bert_pair_with_na.pth.tar')['state_dict']
        else:
            ckpt = torch.load('pretrain/bert_pair/bert_pair_with_na.pth.tar', map_location='cpu')['state_dict']
        sentence_encoder = encoder.BERTPairEncoder(64, 'pretrain/bert-base-uncased')
        model = models.BERTPair(sentence_encoder)
        model.load_state_dict(ckpt)
        model.eval()
        return model
    else:
        raise NotImplementedError

def bag_re_model():
    raise NotImplementedError

def ner_model(name='spacy'):
    if name == 'spacy':
        return models.SpacyNER()
    elif name == 'tagme':
        return models.TagmeNER()
    else:
        raise NotImplementedError