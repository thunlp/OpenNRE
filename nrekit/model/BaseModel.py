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
            (Name of the relation of the sentence, score)
        """
        raise NotImplementedError
    

class BagRE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        raise NotImplementedError

class FewShotRE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def infer(self, support, query):
        """
        Args:
            support: supporting set. 
                [{'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}, 
                  'relation': ...}]
            query: same format as support
        Return:
            [(relation, score), ...]
        """

class NER(nn.Module):
    def __init__(self):
        super().__init__()

    def ner(self, sentence, is_token=False):
        """
        Args:
            sentence: string, the input sentence
            is_token: if is_token == True, senetence becomes an array of token
        Return:
            [{name: xx, pos: [start, end]}], a list of named entities
        """
        raise NotImplementedError
