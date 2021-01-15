import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter


class SentenceEntityRE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 model_name,
                 batch_size=32,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 warmup_step=300,
                 opt='sgd'):

        super().__init__()
        self.max_epoch = max_epoch
        self.softmax = nn.Softmax(-1)
        self.model_name = model_name
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True)
        else:
            self.train_loader = None
        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)
        else:
            self.val_loader = None

        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        else:
            self.test_laoder = None
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
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
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
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits1, logits2, logits3 = self.parallel_model(*args)
                logits = torch.log(self.softmax(logits1 + logits2 + logits3))
                loss1 = self.criterion(logits, label)
                loss2 = self.criterion(logits2, label)
                loss3 = self.criterion(logits3, label)
                loss = loss1 + loss2 + loss3
                score, pred = logits.max(-1)  # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val
            logging.info("=== Epoch %d val ===" % epoch)
            if self.val_loader is not None:
                result = self.eval_model(self.val_loader)
                logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            else:
                result = self.eval_model(self.test_loader)
                logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result_our = []
        pred_result_entity = []
        pred_result_te = []
        pred_result_nde = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                # entity, all, context
                logits1, logits2, logits3 = self.parallel_model(*args)
                score_ent, pred_ent = logits1.max(-1)
                logits = torch.log(self.softmax(logits1 + logits2 + logits3))
                score_te, pred_te = logits.max(-1)
                logits_m = torch.log(self.softmax(logits1 + torch.ones(logits2.size()).to(device='cuda:0') + torch.ones(logits3.size()).to(device='cuda:0')))
                score_nde, pred_nde = logits_m.max(-1)
                logits = self.softmax(logits) - self.softmax(logits_m)
                score_our, pred_our = logits.max(-1)  # (B)
                # Save result
                for i in range(pred_our.size(0)):
                    pred_result_our.append(pred_our[i].item())
                    pred_result_te.append(pred_te[i].item())
                    pred_result_nde.append(pred_nde[i].item())
                    pred_result_entity.append(pred_ent[i].item())
                # Log
                acc = float((pred_our == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred_our.size(0))
                t.set_postfix(acc=avg_acc.avg)
        # 将我们的与base修正比较
        # eval_loader.dataset.write_diff(pred_result, pred_result_o)
        # 只用entity看每个relation结果
        # eval_loader.dataset.eval_entity_model(pred_result_entity)
        eval_loader.dataset.eval_our_model(pred_result_our, self.model_name)
        eval_loader.dataset.eval_te_model(pred_result_te, self.model_name)
        eval_loader.dataset.eval_nde_model(pred_result_nde, self.model_name)
        # self.train_loader.dataset.get_relation_cnt('semeval_rel_num_train.txt')
        # eval_loader.dataset.get_relation_cnt('semeval_rel_num_test.txt')
        result = eval_loader.dataset.eval2(pred_result_our)
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

