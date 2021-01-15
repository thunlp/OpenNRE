import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
from tqdm import tqdm
import json

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data), len(self.rel2id)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        res = [self.rel2id[item['relation']]] + seq
        return [self.rel2id[item['relation']]] + seq # label, seq1, seq2, ...
    
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0)) # (B, L)
        return [batch_labels] + batch_seqs
    
    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]
            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive +=1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0
        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result correct: {}, total: {}, correct_positive: {}, pred_positive: {}, gold_positive: {}'.format(correct, total, correct_positive, pred_positive, gold_positive))

        logging.info('Evaluation result: {}.'.format(result))
        return result

    def eval2(self, pred_result):
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        rel_acc = {}
        rel_tot = {}
        # print(self.rel2id)
        for value in self.rel2id.values():
            rel_acc[value] = 0
            rel_tot[value] = 0
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                neg = self.rel2id[name]
                break
        for i in range(total):
            golden = self.rel2id[self.data[i]['relation']]
            rel_tot[golden] = rel_tot[golden] + 1
            if golden == pred_result[i]:
                rel_acc[golden] = rel_acc[golden] + 1
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        acc2 = 0.0
        for key in rel_acc.keys():
            if rel_tot[key] == 0:
                print('Total number of key {} is 0!'.format(key))
                continue
            acc2 = acc2 + rel_acc[key] / rel_tot[key]
            # logging.info('Acc for relation {} is: {}'.format(key, rel_acc[key]/rel_tot[key]))
        acc2 = acc2 / len(rel_acc.keys())
        # print('Rel_total:', rel_tot)
        logging.info('Acc mean of realtions is: {}'.format(acc2))

        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0
        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result correct: {}, total: {}, correct_positive: {}, pred_positive: {}, gold_positive: {}'.format(correct, total, correct_positive, pred_positive, gold_positive))
        logging.info('Evaluation result: {}.'.format(result))
        return result

    def eval_entity_model(self, pred, model_name):
        total = len(self.data)
        rel_acc = {}
        rel_tot = {}
        for value in self.rel2id.values():
            rel_acc[value] = 0
            rel_tot[value] = 0
        for i in range(total):
            golden = self.rel2id[self.data[i]['relation']]
            rel_tot[golden] = rel_tot[golden] + 1
            if golden == pred[i]:
                rel_acc[golden] = rel_acc[golden] + 1
        for value in self.rel2id.values():
            rel_acc[value] = rel_acc[value] / rel_tot[value]
        with open('../output/{}_acc_pre_relation.txt'.format(model_name), 'w') as f:
            f.write(json.dumps(rel_acc))
        return

    def eval_our_model(self, pred, model_name):
        total = len(self.data)
        rel_acc = {}
        rel_tot = {}
        for value in self.rel2id.values():
            rel_acc[value] = 0
            rel_tot[value] = 0
        for i in range(total):
            golden = self.rel2id[self.data[i]['relation']]
            rel_tot[golden] = rel_tot[golden] + 1
            if golden == pred[i]:
                rel_acc[golden] = rel_acc[golden] + 1
        for value in self.rel2id.values():
            rel_acc[value] = rel_acc[value] / rel_tot[value]
        with open('../output/{}_acc_pre_relation_ourmodel.txt'.format(model_name), 'w') as f:
            f.write(json.dumps(rel_acc))
        return

    def eval_te_model(self, pred, model_name):
        total = len(self.data)
        rel_acc = {}
        rel_tot = {}
        for value in self.rel2id.values():
            rel_acc[value] = 0
            rel_tot[value] = 0
        for i in range(total):
            golden = self.rel2id[self.data[i]['relation']]
            rel_tot[golden] = rel_tot[golden] + 1
            if golden == pred[i]:
                rel_acc[golden] = rel_acc[golden] + 1
        for value in self.rel2id.values():
            rel_acc[value] = rel_acc[value] / rel_tot[value]
        with open('../output/{}_acc_pre_relation_te.txt'.format(model_name), 'w') as f:
            f.write(json.dumps(rel_acc))
        return

    def eval_nde_model(self, pred, model_name):
        total = len(self.data)
        rel_acc = {}
        rel_tot = {}
        for value in self.rel2id.values():
            rel_acc[value] = 0
            rel_tot[value] = 0
        for i in range(total):
            golden = self.rel2id[self.data[i]['relation']]
            rel_tot[golden] = rel_tot[golden] + 1
            if golden == pred[i]:
                rel_acc[golden] = rel_acc[golden] + 1
        for value in self.rel2id.values():
            rel_acc[value] = rel_acc[value] / rel_tot[value]
        with open('../output/{}_acc_pre_relation_nde.txt'.format(model_name), 'w') as f:
            f.write(json.dumps(rel_acc))
        return

    def write_diff(self, pred1, pred2):
        write_file1 = '../output/result_wrong_correct_fix_wiki80.txt'
        write_file2 = '../output/result_correct_wrong_fix_wiki80.txt'

        logging.info('Size of pred: {}'.format(len(pred1)))
        logging.info('Size of label: {}'.format(len(self.data)))
        cnt1 = 0
        cnt2 = 0
        with open(write_file1, 'w') as f1, open(write_file2, 'w') as f2:
            for i in tqdm(range(len(pred1))):
                golden = self.rel2id[self.data[i]['relation']]
                if pred1[i] == golden and pred2[i] != golden:
                    cnt1 = cnt1 + 1
                    self.data[i]['pred1'] = pred1[i]
                    self.data[i]['pred2'] = pred2[i]
                    f1.write(json.dumps(self.data[i]))
                    f1.write('\n')
                elif pred2[i] == golden and pred1[i] != golden:
                    cnt2 = cnt2 + 1
                    self.data[i]['pred1'] = pred1[i]
                    self.data[i]['pred2'] = pred2[i]
                    f2.write(json.dumps(self.data[i]))
                    f2.write('\n')
        logging.info('File1 cnt is: {}'.format(cnt1))
        logging.info('File2 cnt is: {}'.format(cnt2))
        return

    def get_relation_cnt(self, pt):
        rel_tot = {}
        for key in self.rel2id.keys():
            rel_tot[key] = 0
        for dt in self.data:
            rel_tot[dt['relation']] += 1
        with open(os.path.join('../output/', pt), 'w') as f:
            f.write(json.dumps(rel_tot))
        return




def SentenceRELoader(path, rel2id, tokenizer, batch_size,
        shuffle=False, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = SentenceREDataset(path = path, rel2id = rel2id, tokenizer = tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

class BagREDataset(data.Dataset):
    """
    Bag-level relation extraction dataset. Note that relation of NA should be named as 'NA'.
    """
    def __init__(self, path, rel2id, tokenizer, entpair_as_bag=False, bag_size=0, mode=None):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            entpair_as_bag: if True, bags are constructed based on same
                entity pairs instead of same relation facts (ignoring 
                relation labels)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.entpair_as_bag = entpair_as_bag
        self.bag_size = bag_size

        # Load the file
        f = open(path)
        self.data = []
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()

        # Construct bag-level dataset (a bag contains instances sharing the same relation fact)
        if mode == None:
            self.weight = np.ones((len(self.rel2id)), dtype=np.float32)
            self.bag_scope = []
            self.name2id = {}
            self.bag_name = []
            self.facts = {}
            for idx, item in enumerate(self.data):
                fact = (item['h']['id'], item['t']['id'], item['relation'])
                if item['relation'] != 'NA':
                    self.facts[fact] = 1
                if entpair_as_bag:
                    name = (item['h']['id'], item['t']['id'])
                else:
                    name = fact
                if name not in self.name2id:
                    self.name2id[name] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(name)
                self.bag_scope[self.name2id[name]].append(idx)
                self.weight[self.rel2id[item['relation']]] += 1.0
            self.weight = 1.0 / (self.weight ** 0.05)
            self.weight = torch.from_numpy(self.weight)
        else:
            pass
  
    def __len__(self):
        return len(self.bag_scope)

    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if self.bag_size > 0:
            if self.bag_size <= len(bag):
                resize_bag = random.sample(bag, self.bag_size)
            else:
                resize_bag = bag + list(np.random.choice(bag, self.bag_size - len(bag)))
            bag = resize_bag
            
        seqs = None
        rel = self.rel2id[self.data[bag[0]]['relation']]
        for sent_id in bag:
            item = self.data[sent_id]
            seq = list(self.tokenizer(item))
            if seqs is None:
                seqs = []
                for i in range(len(seq)):
                    seqs.append([])
            for i in range(len(seq)):
                seqs[i].append(seq[i])
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (n, L), n is the size of bag
        return [rel, self.bag_name[index], len(bag)] + seqs
  
    def collate_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.cat(seqs[i], 0) # (sumn, L)
            seqs[i] = seqs[i].expand((torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1, ) + seqs[i].size())
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        assert(start == seqs[0].size(1))
        scope = torch.tensor(scope).long()
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

    def collate_bag_size_fn(data):
        data = list(zip(*data))
        label, bag_name, count = data[:3]
        seqs = data[3:]
        for i in range(len(seqs)):
            seqs[i] = torch.stack(seqs[i], 0) # (batch, bag, L)
        scope = [] # (B, 2)
        start = 0
        for c in count:
            scope.append((start, start + c))
            start += c
        label = torch.tensor(label).long() # (B)
        return [label, bag_name, scope] + seqs

  
    def eval(self, pred_result):
        """
        Args:
            pred_result: a list with dict {'entpair': (head_id, tail_id), 'relation': rel, 'score': score}.
                Note that relation of NA should be excluded.
        Return:
            {'prec': narray[...], 'rec': narray[...], 'mean_prec': xx, 'f1': xx, 'auc': xx}
                prec (precision) and rec (recall) are in micro style.
                prec (precision) and rec (recall) are sorted in the decreasing order of the score.
                f1 is the max f1 score of those precison-recall points
        """
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec) 
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'micro_p': np_prec, 'micro_r': np_rec, 'micro_p_mean': mean_prec, 'micro_f1': f1, 'auc': auc}

def BagRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, entpair_as_bag=False, bag_size=0, num_workers=8, 
        collate_fn=BagREDataset.collate_fn):
    if bag_size == 0:
        collate_fn = BagREDataset.collate_fn
    else:
        collate_fn = BagREDataset.collate_bag_size_fn
    dataset = BagREDataset(path, rel2id, tokenizer, entpair_as_bag=entpair_as_bag, bag_size=bag_size)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader
