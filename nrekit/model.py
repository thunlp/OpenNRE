import torch
from torch import nn
from .framework import SentenceRE
from .sentence_encoder import CNNSentenceEncoder

class CNNSoftmax(SentenceRE):
    def __init__(self, sentence_encoder, num_class, id2rel, hidden_size=230):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
            hidden_size: hidden size of sentence encoder
        """
        super.__init__()
        self.sentence_encoder = sentence_encoder
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.fc = nn.Linear(hidden_size, num_class)
        self.id2rel = id2rel

    @overrides
    def forward(self, sentence, pos_head, pos_tail, is_token=False):
        token, pos1, pos2 = self.sentence_encoder.tokenize(sentence, pos_head, pos_tail, is_token)
        rep = self.sentence_encoder(token, pos1, pos2) # (1, H)
        logits = self.fc(rep) # (1, N)
        score, pred = logits.max(-1) # (1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred]
    