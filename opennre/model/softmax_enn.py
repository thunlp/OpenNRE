import torch
from torch import nn, optim
from .base_model import SentenceRE


class SoftmaxENN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc1 = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.fc2 = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.fc3 = nn.Linear(self.sentence_encoder.hidden_size, num_class)

        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep1, rep2, rep3 = self.sentence_encoder(*args)  # (B, H)
        rep1 = self.drop(rep1)
        rep2 = self.drop(rep2)
        rep3 = self.drop(rep3)
        logits1 = self.fc1(rep1)  # (B, N)
        logits2 = self.fc2(rep2)
        logits3 = self.fc3(rep3)
        return logits1, logits2, logits3
