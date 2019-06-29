import torch
from torch import nn, optim
from ..data_loader import get_bag_re_loader
from ..framework import BagRE
from ..util import *
from tqdm import tqdm
import numpy as np

class PCNN_ATT(BagRE):
    def __init__(self, sentence_encoder, num_class, rel2id, hidden_size=230):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
            hidden_size: hidden size of sentence encoder
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.fc = nn.Linear(hidden_size * 3, num_class)
        self.drop = nn.Dropout()
        self.rel2id = rel2id
        self.id2rel = {}
        self.softmax = nn.Softmax(-1)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            if 'text' in item:
                token, pos1, pos2, mask = self.tokenizer(item['text'], 
                    item['h']['pos'], item['t']['pos'], is_token=False, padding=True)
            else:
                token, pos1, pos2, mask = self.tokenizer(item['token'], 
                    item['h']['pos'], item['t']['pos'], is_token=True, padding=True)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0) # (n, L)
        pos1s = torch.cat(pos1s, 0)
        pos2s = torch.cat(pos2s, 0)
        masks = torch.cat(masks, 0) 
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max()
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)
    
    def forward(self, label, scope, token, pos1, pos2, mask, train=True):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            pos1: (nsum, L), relative position to head entity
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(token, pos1, pos2, mask) # (nsum, H)

        # Attention
        if train:
            bag_rep = []
            query = torch.zeros((rep.size(0))).long()
            if torch.cuda.is_available():
                query = query.cuda()
            for i in range(len(scope)):
                query[scope[i][0]:scope[i][1]] = label[i]
            att_mat = self.fc.weight.data[query] # (nsum, H)
            att_score = (rep * att_mat).sum(-1) # (nsum)
            for i in range(len(scope)):
                bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) # (n)
                bag_rep.append(torch.matmul(softmax_att_score.unsqueeze(0), bag_mat).squeeze(0)) # (1, n) * (n, H) -> (1, H) -> (H)
            bag_rep = torch.stack(bag_rep, 0) # (B, H)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:
            bag_logits = []
            att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
            for i in range(len(scope)):
                bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, (softmax)n) 
                rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel)) # ((each rel)N, (logit)N)
                logit_for_each_rel = logit_for_each_rel.diag() # (N)
                bag_logits.append(logit_for_each_rel)
            bag_logits = torch.stack(bag_logits,0) # after **softmax**

        return bag_logits

    def train_model(self, train_path, val_path, ckpt,
            batch_size=160, max_epoch=60, lr=0.5, 
            weight_decay=0, opt='sgd'):
        """
        Train the model with given data
        Args:
            train_path: path of train data
            val_path: path of val data
            rel2id: dictionary of rel->id mapping
            batch_size: batch size
            max_epoch: max epoch of training
            lr: learning rate
            weight_decay: weight decay
            opt: optimizer. 'sgd' or 'adam'
        """
        # Load data
        print('Reading and processing data...')
        train_loader = get_bag_re_loader(
            train_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            True,
            entpair_as_bag=False
        )
        val_loader = get_bag_re_loader(
            val_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            False,
            entpair_as_bag=True
        )
        print('Finished')
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Params and optimizer
        params = self.parameters()
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam'.")

        # Cuda
        if torch.cuda.is_available():
            self.cuda()

        best_auc = 0
        for epoch in range(max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()
            t = tqdm(train_loader)
            for iter, data in enumerate(t):
                label, bag_name, scope, token, pos1, pos2, mask = data
                if torch.cuda.is_available():
                    token = token.cuda()
                    pos1 = pos1.cuda()
                    pos2 = pos2.cuda()
                    mask = mask.cuda()
                    label = label.cuda()
                logits = self.forward(label, scope, token, pos1, pos2, mask)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)
                
                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Val 
            self.eval()
            print("=== Epoch %d val ===" % epoch)
            with torch.no_grad():
                t = tqdm(val_loader)
                pred_result = []
                for iter, data in enumerate(t):
                    label, bag_name, scope, token, pos1, pos2, mask = data
                    if torch.cuda.is_available():
                        token = token.cuda()
                        pos1 = pos1.cuda()
                        pos2 = pos2.cuda()
                        label = label.cuda()
                        mask = mask.cuda()
                    logits = self.forward(None, scope, token, pos1, pos2, mask, train=False) # results after softmax
                    for i in range(logits.size(0)):
                        for relid in range(self.num_class):
                            if self.id2rel[relid] != 'NA':
                                pred_result.append({
                                    'entpair': bag_name[i][:2], 
                                    'relation': self.id2rel[relid], 
                                    'score': logits[i][relid].item()
                                })
                result = val_loader.dataset.eval(pred_result)
                print("auc: %.4f" % result['auc'])
                print("f1: %.4f" % (result['f1']))

            if result['auc'] > best_auc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.state_dict()}, ckpt)
                best_auc = result['auc']
        print("Best auc on val set: %f" % (best_auc))