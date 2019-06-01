from . import encoder
from . import framework
from . import models

__all__ = ['encoder', 'framework', 'models']

import json
import torch

def sentence_re_model(name='cnn_softmax_wiki64'):
    if name == 'cnn_softmax_wiki64':
        wordi2d = json.load(open('data/word2id.json'))
        id2rel = json.load(open('data/id2rel.json'))
        ckpt = torch.load('data/cnn_encoder.pth.tar')['state_dict']
        sentence_encoder = encoder.CNNEncoder(len(wordi2d), wordi2d, 40)
        model = models.CNNSoftmax(sentence_encoder, len(id2rel), id2rel)
        model.load_state_dict(ckpt)
        model.eval()
        return model
    else:
        raise NotImplementedError

def bag_re_model():
    raise NotImplementedError

def fewshot_re_model():
    raise NotImplementedError