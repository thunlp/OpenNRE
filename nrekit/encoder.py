import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from nltk import word_tokenize

class CNNEncoder(nn.Module):

    def __init__(self, num_word, word2id, max_length, 
        word_embedding_dim=50, pos_embedding_dim=5, kernel_size=3, padding=1, hidden_size=230):
        """
        Args:
            num_word: number of words, including 'UNK' and 'BLANK'
            word2id: dictionary of word->idx mapping
            max_length: max length of sentence, used for postion embedding
            word_embedding_dim: dimention of word embedding
            pos_embedding_dim: dimention of position embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
            hidden_size: hidden size
        """
        super().__init__()
        self.num_word = num_word
        self.word2id = word2id
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.hidden_size = hidden_size
    
        # Word embedding
        self.word_embedding = nn.Embedding(num_word, self.word_embedding_dim)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

        # CNN
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, kernel_size, padding=padding)

    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, H), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1), 
                       self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.conv(x) # (B, H, L)
        x = torch.relu(x) # (B, H, L)
        x, _ = x.max(-1) # (B, H)
        return x

    def tokenize(self, sentence, pos_head, pos_tail, is_token=False):
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
        if not is_token:
            tokens = word_tokenize(sentence)
            ori_pos_head = pos_head.copy()
            ori_pos_tail = pos_tail.copy()
            pos_head = []
            pos_tail = []
            cur_pos = 0
            for i in range(len(tokens)):
                cur_pos = sentence.find(tokens[i], cur_pos)
                if ori_pos_head[0] <= cur_pos and cur_pos < ori_pos_head[1]:
                    pos_head.append(i)
                if ori_pos_tail[0] <= cur_pos and cur_pos < ori_pos_tail[1]:
                    pos_tail.append(i)
                cur_pos += len(tokens[i])
            if len(pos_head) == 0 or len(pos_tail) == 0:
                raise Exception("Cannot locate head or tail entity!")
            pos_head = [pos_head[0], pos_head[-1] + 1]
            pos_tail = [pos_tail[0], pos_tail[-1] + 1]
        else:
            tokens = sentence
        
        # Token -> index
        indexed_tokens = []
        for token in tokens:
            # Not case-sensitive
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['UNK'])

        # Position
        pos1 = []
        pos2 = []
        pos1_in_index = pos_head[0]
        pos2_in_index = pos_tail[0]
        for i in range(len(indexed_tokens)):
            pos1.append(i - pos1_in_index + self.max_length)
            pos2.append(i - pos2_in_index + self.max_length)

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0) # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0) # (1, L)

        return indexed_tokens, pos1, pos2
