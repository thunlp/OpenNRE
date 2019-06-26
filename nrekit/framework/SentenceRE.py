import torch
from torch import nn, optim
import json
from .DataLoader import SentenceRELoader
from .utils import AverageMeter
from tqdm import tqdm

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 ckpt, 
                 batch_size = 32, 
                 max_epoch = 100, 
                 lr = 0.1, 
                 weight_decay = 1e-5, 
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
        # Model
        self.model = model
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
        # ckpt
        self.ckpt = ckpt

    def train_model(self):
        best_acc = 0
        for epoch in range(self.max_epoch):
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                label, token, pos1, pos2 = data
                if torch.cuda.is_available():
                    token = token.cuda()
                    pos1 = pos1.cuda()
                    pos2 = pos2.cuda()
                    label = label.cuda()
                logits = self.model(token, pos1, pos2)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                if iter % 10 == 0:
                    t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Val 
            self.eval()
            print("=== Epoch %d val ===" % epoch)
            avg_acc = AverageMeter()
            with torch.no_grad():
                t = tqdm(self.val_loader)
                for iter, data in enumerate(t):
                    label, token, pos1, pos2 = data
                    if torch.cuda.is_available():
                        token = token.cuda()
                        pos1 = pos1.cuda()
                        pos2 = pos2.cuda()
                        label = label.cuda()
                    logits = self.model(token, pos1, pos2)
                    score, pred = logits.max(-1) # (B)
                    acc = float((pred == label).long().sum()) / label.size(0)
                    # Log
                    avg_acc.update(acc, 1)
                    if iter % 10 == 0:
                        t.set_postfix(acc=avg_acc.avg)
            if avg_acc.avg > best_acc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.state_dict()}, self.ckpt)
                best_acc = avg_acc.avg
        print("Best acc on val set: %f" % (best_acc))
