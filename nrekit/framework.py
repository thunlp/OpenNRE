import torch
from torch import nn
import json

class SentenceRE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def infer(self, sentence, pos_head, pos_tail, is_token=False):
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
