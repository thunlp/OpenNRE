import torch
from torch import nn, optim
from ..data_loader import get_sentence_re_loader
from ..framework import SentenceRE
from ..util import *
from tqdm import tqdm

class CNNSoftmax(SentenceRE):
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
        self.fc = nn.Linear(hidden_size, num_class)
        self.drop = nn.Dropout()
        self.rel2id = rel2id
        self.id2rel = {}
        self.softmax = nn.Softmax(-1)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, sentence, pos_head, pos_tail, is_token=False):
        token, pos1, pos2 = self.sentence_encoder.tokenize(sentence, pos_head, pos_tail, is_token)
        logits = self.forward(token, pos1, pos2)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(token, pos1, pos2) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep) # (B, N)
        return logits

    def train_model(self, train_path, val_path, ckpt,
            batch_size=32, max_epoch=100, lr=0.1, 
            weight_decay=1e-5, opt='sgd'):
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
        train_loader = get_sentence_re_loader(
            train_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            True)
        val_loader = get_sentence_re_loader(
            val_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            False)
        
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

        best_acc = 0
        for epoch in range(max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(train_loader)
            for iter, data in enumerate(t):
                label, token, pos1, pos2 = data
                if torch.cuda.is_available():
                    token = token.cuda()
                    pos1 = pos1.cuda()
                    pos2 = pos2.cuda()
                    label = label.cuda()
                logits = self.forward(token, pos1, pos2)
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
                t = tqdm(val_loader)
                for iter, data in enumerate(t):
                    label, token, pos1, pos2 = data
                    if torch.cuda.is_available():
                        token = token.cuda()
                        pos1 = pos1.cuda()
                        pos2 = pos2.cuda()
                        label = label.cuda()
                    logits = self.forward(token, pos1, pos2)
                    score, pred = logits.max(-1) # (B)
                    acc = float((pred == label).long().sum()) / label.size(0)

                    # Log
                    avg_acc.update(acc, 1)
                    if iter % 10 == 0:
                        t.set_postfix(acc=avg_acc.avg)
            if avg_acc.avg > best_acc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.state_dict()}, ckpt)
                best_acc = avg_acc.avg
        print("Best acc on val set: %f" % (best_acc))

class BERTSoftmax(SentenceRE):
    def __init__(self, sentence_encoder, num_class, rel2id, hidden_size=768):
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
        self.fc = nn.Linear(hidden_size, num_class)
        self.drop = nn.Dropout()
        self.rel2id = rel2id
        self.id2rel = {}
        self.softmax = nn.Softmax(-1)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, sentence, pos_head, pos_tail, is_token=False):
        token, att_mask = self.sentence_encoder.tokenize(sentence, pos_head, pos_tail, is_token)
        logits = self.forward(token, att_mask)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(token, att_mask) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep) # (B, N)
        return logits

    def train_model(self, train_path, val_path, ckpt,
            batch_size=32, max_epoch=10, lr=2e-5):
        """
        Train the model with given data
        Args:
            train_path: path of train data
            val_path: path of val data
            rel2id: dictionary of rel->id mapping
            batch_size: batch size
            max_epoch: max epoch of training
            lr: learning rate
        """
        # Load data
        train_loader = get_sentence_re_loader(
            train_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            True)
        val_loader = get_sentence_re_loader(
            val_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            False)
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Params and optimizer
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
        from pytorch_pretrained_bert import BertAdam
        self.optimizer = BertAdam(grouped_params)

        # Cuda
        if torch.cuda.is_available():
            self.cuda()

        best_acc = 0
        global_step = 0
        for epoch in range(max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(train_loader)
            for iter, data in enumerate(t):
                label, token, att_mask = data
                if torch.cuda.is_available():
                    token = token.cuda()
                    att_mask = att_mask.cuda()
                    label = label.cuda()
                logits = self.forward(token, att_mask) 
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                
                # Optimize
                ## Warm-up
                def warmup_linear(global_step, warmup_step=300):
                    if global_step < warmup_step:
                        return float(global_step) / warmup_step
                    else:
                        return 1.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['ori_lr'] * warmup_linear(global_step)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # global step
                global_step += 1

            # Val 
            self.eval()
            print("=== Epoch %d val ===" % epoch)
            avg_acc = AverageMeter()
            with torch.no_grad():
                t = tqdm(val_loader)
                for iter, data in enumerate(t):
                    label, token, att_mask = data
                    if torch.cuda.is_available():
                        token = token.cuda()
                        att_mask = att_mask.cuda()
                        label = label.cuda()
                    logits = self.forward(token, att_mask)
                    score, pred = logits.max(-1) # (B)
                    acc = float((pred == label).long().sum()) / label.size(0)

                    # Log
                    avg_acc.update(acc, 1)
                    if iter % 10 == 0:
                        t.set_postfix(acc=avg_acc.avg)
            if avg_acc.avg > best_acc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.state_dict()}, ckpt)
                best_acc = avg_acc.avg
        print("Best acc on val set: %f" % (best_acc))

class MTBSoftmax(SentenceRE):
    def __init__(self, sentence_encoder, num_class, rel2id, hidden_size=768*2):
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
        self.fc = nn.Linear(hidden_size, num_class)
        self.drop = nn.Dropout()
        self.rel2id = rel2id
        self.id2rel = {}
        self.softmax = nn.Softmax(-1)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, sentence, pos_head, pos_tail, is_token=False):
        token, att_mask, pos1, pos2 = self.sentence_encoder.tokenize(sentence, pos_head, pos_tail, is_token)
        logits = self.forward(token, att_mask, pos1, pos2)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(token, att_mask, pos1, pos2) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep) # (B, N)
        return logits

    def train_model(self, train_path, val_path, ckpt,
            batch_size=32, max_epoch=10, lr=2e-5):
        """
        Train the model with given data
        Args:
            train_path: path of train data
            val_path: path of val data
            rel2id: dictionary of rel->id mapping
            batch_size: batch size
            max_epoch: max epoch of training
            lr: learning rate
        """
        # Load data
        train_loader = get_sentence_re_loader(
            train_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            True)
        val_loader = get_sentence_re_loader(
            val_path,
            self.rel2id,
            self.sentence_encoder.tokenize,
            batch_size,
            False)
        
        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Params and optimizer
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
        from pytorch_pretrained_bert import BertAdam
        self.optimizer = BertAdam(grouped_params)

        # Cuda
        if torch.cuda.is_available():
            self.cuda()

        best_acc = 0
        global_step = 0
        for epoch in range(max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(train_loader)
            for iter, data in enumerate(t):
                label, token, att_mask, pos1, pos2 = data
                if torch.cuda.is_available():
                    token = token.cuda()
                    att_mask = att_mask.cuda()
                    pos1 = pos1.cuda()
                    pos2 = pos2.cuda()
                    label = label.cuda()
                logits = self.forward(token, att_mask) 
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)

                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                
                # Optimize
                ## Warm-up
                def warmup_linear(global_step, warmup_step=300):
                    if global_step < warmup_step:
                        return float(global_step) / warmup_step
                    else:
                        return 1.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['ori_lr'] * warmup_linear(global_step)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # global step
                global_step += 1

            # Val 
            self.eval()
            print("=== Epoch %d val ===" % epoch)
            avg_acc = AverageMeter()
            with torch.no_grad():
                t = tqdm(val_loader)
                for iter, data in enumerate(t):
                    label, token, att_mask, pos1, pos2 = data
                    if torch.cuda.is_available():
                        token = token.cuda()
                        att_mask = att_mask.cuda()
                        pos1 = pos1.cuda()
                        pos2 = pos2.cuda()
                        label = label.cuda()
                    logits = self.forward(token, att_mask, pos1, pos2)
                    score, pred = logits.max(-1) # (B)
                    acc = float((pred == label).long().sum()) / label.size(0)

                    # Log
                    avg_acc.update(acc, 1)
                    if iter % 10 == 0:
                        t.set_postfix(acc=avg_acc.avg)
            if avg_acc.avg > best_acc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.state_dict()}, ckpt)
                best_acc = avg_acc.avg
        print("Best acc on val set: %f" % (best_acc))