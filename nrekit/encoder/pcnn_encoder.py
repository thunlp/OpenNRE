import torch
import torch.nn as nn

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder

class PCNNEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, 
                 max_length=128, 
                 hidden_size=230, 
                 word_size=50,
                 position_size=5,
                 padding=True,
                 word2vec=None,
                 kernel_size=3, 
                 padding_size=1,
                 dropout=0.5):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            padding: padding for CNN
            word2vec: pretrained word2vec numpy
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # hyperparameters
        super().__init__(token2id, max_length, hidden_size, word_size, position_size, padding, word2vec)
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.padding_size = padding_size

        self.conv = CNN(self.input_size, self.hidden_size, self.dropout, self.kernel_size, self.padding_size, activation_function = F.relu())
        self.pool = MaxPool(self.max_length, 3)

    def forward(self, token, pos1, pos2, mask):
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
        x = self.conv(x) # (B, L, EMBED)
        x = self.pool(x) # (B, EMBED)
        return x

    def tokenize(self, item):
        """
        Args:
            sentence: string, the input sentence
            pos_head: [start, end], position of the head entity
            pos_end: [start, end], position of the tail entity
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        indexed_tokens, pos1, pos2 = super.tokenize(tokenize, is_token)
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']        

        # Mask
        pos1_in_index = min(pos_head[0], self.max_length)
        pos2_in_index = min(pos_tail[0], self.max_length)

        mask = []
        pos_min = min(pos1_in_index, pos2_in_index)
        pos_max = max(pos1_in_index, pos2_in_index)
        for i in range(len(indexed_tokens)):
            if i <= pos_min:
                mask.append(1)
            elif i <= pos_max:
                mask.append(2)
            else:
                mask.append(3)
        # Padding
        if self.padding:
            while len(mask) < self.max_length:
                mask.append(0)
            mask = mask[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0) # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0) # (1, L)
        mask = torch.tensor(mask).long().unsqueeze(0) # (1, L)
        return indexed_tokens, pos1, pos2, mask
