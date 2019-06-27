import torch
import torch.nn as nn

from .BaseEncoder import BaseEncoder
from pytorch_pretrained_bert import BertModel, BertTokenizer

class BERTEncoder(BaseEncoder):

    def __init__(self, max_length, pretrain_path):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask)
        return x

    def tokenize(self, item, is_token = False):
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
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        else:
            tokens = sentence
        
        # Token -> index
        re_tokens = ['[CLS]']
        cur_pos = 0
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                re_tokens.append('[HEADSTART]')
            if cur_pos == pos_tail[0]:
                re_tokens.append('[TAILSTART]')
            re_tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[HEADEND]')
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[TAILEND]')
            cur_pos += 1

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if self.padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0) # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long() # (1, L)
        att_mask[0, :avai_len] = 1
        
        return indexed_tokens, att_mask
