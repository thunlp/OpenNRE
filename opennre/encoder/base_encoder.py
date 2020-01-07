import math, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..tokenization import WordTokenizer

class BaseEncoder(nn.Module):

    def __init__(self, 
                 token2id, 
                 max_length=128, 
                 hidden_size=230, 
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
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
        """
        # Hyperparameters
        super().__init__()

        self.token2id = token2id
        self.max_length = max_length
        self.num_token = len(token2id)
        self.num_position = max_length * 2
        self.mask_entity = mask_entity

        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]
            
        self.position_size = position_size
        self.hidden_size = hidden_size
        self.input_size = word_size + position_size * 2
        self.blank_padding = blank_padding

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1

        # Word embedding
        self.word_embedding = nn.Embedding(self.num_token, self.word_size)
        if word2vec is not None:
            logging.info("Initializing word embedding with word2vec.")
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:            
                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                blk = torch.zeros(1, self.word_size)
                self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
            else:
                self.word_embedding.weight.data.copy_(word2vec)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]")

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
        pass
    
    def tokenize(self, item):
        """
        Args:
            item: input instance, including sentence, entity positions, etc.
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            index number of tokens and positions             
        """
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        # Sentence -> token
        if not is_token:
            if pos_head[0] > pos_tail[0]:
                pos_min, pos_max = [pos_tail, pos_head]
                rev = True
            else:
                pos_min, pos_max = [pos_head, pos_tail]
                rev = False
            sent_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            sent_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            sent_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            if self.mask_entity:
                ent_0 = ['[UNK]']
                ent_1 = ['[UNK]']
            tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
            if rev:
                pos_tail = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_head = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
            else:
                pos_head = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_tail = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]           
        else:
            tokens = sentence

        # Token -> index
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'], self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.token2id['[UNK]'])

        # Position -> index
        pos1 = []
        pos2 = []
        pos1_in_index = min(pos_head[0], self.max_length)
        pos2_in_index = min(pos_tail[0], self.max_length)
        for i in range(len(tokens)):
            pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
            pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))

        if self.blank_padding:                
            while len(pos1) < self.max_length:
                pos1.append(0)
            while len(pos2) < self.max_length:
                pos2.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
            pos1 = pos1[:self.max_length]
            pos2 = pos2[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0) # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0) # (1, L)
        
        return indexed_tokens, pos1, pos2
