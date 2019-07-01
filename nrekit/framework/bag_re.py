import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader
from .utils import AverageMeter
from tqdm import tqdm
import os

class BagRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd'):
    
        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'bert_adam':
            from pytorch_pretrained_bert import BertAdam
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = BertAdam(grouped_params)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self):
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
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("auc: %.4f" % result['auc'])
            print("f1: %.4f" % (result['f1']))
            if result['auc'] > best_auc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.state_dict()}, ckpt)
                best_auc = result['auc']
        print("Best auc on val set: %f" % (best_auc))

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
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
            result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.parallel_model.load_state_dict(state_dict)