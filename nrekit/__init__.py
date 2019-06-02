from . import encoder
from . import framework
from . import models

__all__ = ['encoder', 'framework', 'models']

import json
import torch

def sentence_re_model(name='cnn_softmax_wiki64'):
    if name == 'cnn_softmax_wiki64':
        wordi2d = json.load(open('data/word2id.json'))
        rel2id = json.load(open('data/wiki80_rel2id.json'))
        ckpt = torch.load('ckpt/wiki80_cnn.pth.tar')['state_dict']
        sentence_encoder = encoder.CNNEncoder(len(wordi2d), wordi2d, 40)
        model = models.CNNSoftmax(sentence_encoder, len(rel2id), rel2id)
        model.load_state_dict(ckpt)
        model.eval()
        return model
    else:
        raise NotImplementedError

def bag_re_model():
    raise NotImplementedError

def fewshot_re_model():
    raise NotImplementedError