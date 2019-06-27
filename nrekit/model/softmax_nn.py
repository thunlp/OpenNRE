import torch
from torch import nn, optim
from .BaseModel import SentenceRE

class SoftmaxNN(SentenceRE):

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
            hidden_size: hidden size of sentence encoder
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item, is_token = False):
        token, pos1, pos2 = self.sentence_encoder.tokenize(item, is_token)
        logits = self.forward(token, pos1, pos2)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(token, pos1, pos2) # (B, H)
        logits = self.fc(rep) # (B, N)
        return logits