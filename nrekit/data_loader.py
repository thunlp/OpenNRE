import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        if 'text' in item:
            token, pos1, pos2 = self.tokenizer(item['text'], 
                item['h']['pos'], item['t']['pos'], is_token=False, padding=True)
        else:
            token, pos1, pos2 = self.tokenizer(item['token'], 
                item['h']['pos'], item['t']['pos'], is_token=True, padding=True)
        return token, pos1, pos2, self.rel2id[item['relation']]
    
    def collate_fn(data):
        token, pos1, pos2, label = zip(*data)
        token = torch.cat(token, 0) # (B, L)
        pos1 = torch.cat(pos1, 0) # (B, L)
        pos2 = torch.cat(pos2, 0) # (B, L)
        label = torch.tensor(label).long() # (B)
        return token, pos1, pos2, label
    
def get_sentence_re_loader(path, rel2id, tokenizer, batch_size, 
        shuffle, num_workers=4, collate_fn=SentenceREDataset.collate_fn):
    dataset = SentenceREDataset(path, rel2id, tokenizer)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

