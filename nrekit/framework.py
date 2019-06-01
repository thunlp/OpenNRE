import torch
from torch import nn

class SentenceRE(nn.Module):
    def __init__(self):
        super.__init__()
    
    def forward(self, sentence, pos_head, pos_tail, is_token=False):
        """
        Args:
            sentence: string, the input sentence
            pos_head: [start, end], position of the head entity
            pos_tail: [start, end], position of the tail entity
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            Name of the relation of the sentence
        """
        raise NotImplementedError

def sentence_re_model():
    return None

def bag_re_model():
    raise NotImplementedError

def fewshot_re_model():
    raise NotImplementedError