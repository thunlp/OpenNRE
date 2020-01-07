import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder

class CNNEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, 
                 max_length=128, 
                 hidden_size=230, 
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3, 
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # Hyperparameters
        super(CNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1), 
                       self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.act(self.conv(x)) # (B, H, L)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)
