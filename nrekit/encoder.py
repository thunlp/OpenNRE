import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from nltk import word_tokenize

class CNNEncoder(nn.Module):

    def __init__(self, num_word, word2id, max_length, 
        word_embedding_dim=50, pos_embedding_dim=5, kernel_size=3, padding=1, hidden_size=230, word2vec=None):
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
            word2vec: pretrained word2vec numpy
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
        if word2vec is not None:
            print("Initializing word embedding with word2vec...")
            word2vec = torch.from_numpy(word2vec)
            unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
            blk = torch.zeros(1, word_embedding_dim)
            self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
            print('Finished')

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

    def tokenize(self, sentence, pos_head, pos_tail, is_token=False, padding=False):
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
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = word_tokenize(sentence[:pos_min[0]])
            ent0 = word_tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = word_tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = word_tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = word_tokenize(sentence[pos_max[1]:])
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
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
        pos1_in_index = min(pos_head[0], self.max_length)
        pos2_in_index = min(pos_tail[0], self.max_length)
        for i in range(len(indexed_tokens)):
            pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
            pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))

        # Padding
        if padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(self.word2id['BLANK'])
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

class PCNNEncoder(nn.Module):

    def __init__(self, num_word, word2id, max_length, 
        word_embedding_dim=50, pos_embedding_dim=5, kernel_size=3, padding=1, hidden_size=230, word2vec=None):
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
            word2vec: pretrained word2vec numpy
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
        if word2vec is not None:
            print("Initializing word embedding with word2vec")
            word2vec = torch.from_numpy(word2vec)
            unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
            blk = torch.zeros(1, word_embedding_dim)
            self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

        # PCNN
        self.pool = nn.MaxPool1d(self.max_length)
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, kernel_size, padding=padding)
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

    def forward(self, token, pos1, pos2, mask):
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

        mask = 1 - self.mask_embedding(mask).transpose(1, 2) # (B, L) -> (B, L, 3) -> (B, 3, L)
        pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :])) # (B, H, 1)
        pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
        x = torch.cat([pool1, pool2, pool3], 1) # (B, 3H, 1)
        x = x.squeeze(2) # (B, 3H)

        return x

    def tokenize(self, sentence, pos_head, pos_tail, is_token=False, padding=False):
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
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = word_tokenize(sentence[:pos_min[0]])
            ent0 = word_tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = word_tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = word_tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = word_tokenize(sentence[pos_max[1]:])
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [len(sent0) + len(ent0) + len(sent1), len(sent0) + len(ent0) + len(sent1) + len(ent1)]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
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
        pos1_in_index = min(pos_head[0], self.max_length)
        pos2_in_index = min(pos_tail[0], self.max_length)
        for i in range(len(indexed_tokens)):
            pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
            pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))

        # Mask
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
        if padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(self.word2id['BLANK'])
            while len(pos1) < self.max_length:
                pos1.append(0)
            while len(pos2) < self.max_length:
                pos2.append(0)
            while len(mask) < self.max_length:
                mask.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
            pos1 = pos1[:self.max_length]
            pos2 = pos2[:self.max_length]
            mask = mask[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0) # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0) # (1, L)
        mask = torch.tensor(mask).long().unsqueeze(0) # (1, L)

        return indexed_tokens, pos1, pos2, mask

class BERTEncoder(nn.Module):

    def __init__(self, max_length, pretrain_path):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        from pytorch_pretrained_bert import BertTokenizer, BertModel
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

    def tokenize(self, sentence, pos_head, pos_tail, is_token=False, padding=False):
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
        re_tokens.append("[SEP]")

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0) # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long() # (1, L)
        att_mask[0, :avai_len] = 1
        
        return indexed_tokens, att_mask

class BERTPairEncoder(nn.Module):

    def __init__(self, max_length, pretrain_path):
        """
        Args:
            max_length: max length of one sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length * 2
        from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
        self.bert = BertForSequenceClassification.from_pretrained(pretrain_path, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, token, seg, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            seg: (B, L), segment
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, 2), logits
        """
        x = self.bert(token, seg, attention_mask=att_mask)
        return x

    def _tokenize_single(self, sentence, pos_head, pos_tail, is_token=False):
        """
        Args:
            sentence: string, the input sentence
            pos_head: [start, end], position of the head entity
            pos_tail: [start, end], position of the tail entity
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            indexed_token
        """
        # Sentence -> token
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
        re_tokens = []
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
        return indexed_tokens
    
    def tokenize(self, sentence1, pos_head1, pos_tail1, sentence2, pos_head2, pos_tail2, is_token=False, padding=False):
        """
        Args:
            sentence1: string, the input sentence
            pos_head1: [start, end], position of the head entity
            pos_tail2: [start, end], position of the tail entity
            sentence2, pos_head2, pos_tail2: same as above
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            indexed_token, seg, att_mask
        """
        token1 = self._tokenize_single(sentence1, pos_head1, pos_tail1, is_token)
        token2 = self._tokenize_single(sentence2, pos_head2, pos_tail2, is_token)
        CLS = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        SEP = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        pair_token = [CLS] + token1 + [SEP] + token2 + [SEP]

        # Padding
        avai_len = len(pair_token)
        if padding:
            while len(pair_token) < self.max_length:
                pair_token.append(0) # 0 is id for [PAD]
            pair_token = pair_token[:self.max_length]
        pair_token = torch.tensor(pair_token).long().unsqueeze(0) # (1, L)

        # Attention mask
        att_mask = torch.zeros(pair_token.size()).long() # (1, L)
        att_mask[0, :avai_len] = 1

        # Sep
        sep = torch.ones(pair_token.size()).long() # (1, L)
        sep[0, :min(self.max_length, len(token1) + 1)] = 0
        
        return pair_token, sep, att_mask

class MTB(nn.Module):
    """
    Matching the blank.
    """

    def __init__(self, max_length, pretrain_path):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.bert(token, attention_mask=att_mask)
        hidden = hidden[-1] # the last hidden layer, (B, L, H)
        # Get entity start hidden state
        onehot = torch.zeros(hidden.size()[:2]).long() # (B, L)
        if torch.cuda.is_available():
            onehot = onehot.cuda()
        onehot_head = onehot.scatter_(1, pos1, 1)
        onehot_tail = onehot.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1) # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1) # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1) # (B, 2H)
        return x

    def tokenize(self, sentence, pos_head, pos_tail, is_token=False, padding=False):
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
        pos1 = 0
        pos2 = 0
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1 = len(re_tokens)
                re_tokens.append('[HEADSTART]')
            if cur_pos == pos_tail[0]:
                pos2 = len(re_tokens)
                re_tokens.append('[TAILSTART]')
            re_tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[HEADEND]')
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[TAILEND]')
            cur_pos += 1
        re_tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0) # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long() # (1, L)
        att_mask[0, :avai_len] = 1
        
        return indexed_tokens, att_mask, pos1, pos2