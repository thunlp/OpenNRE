import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics

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
        logging.info('Evaluation result: {}.'.format(result))
        return result
    
def SentenceRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
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
                # Annotated test set
                if 'anno_relation_list' in item:
                    for r in item['anno_relation_list']:
                        fact = (item['h']['id'], item['t']['id'], r)
                        if r != 'NA':
                            self.facts[fact] = 1
                    assert entpair_as_bag
                    name = (item['h']['id'], item['t']['id'])
                else:
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

  
    def eval(self, pred_result, threshold=0.5):
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

        entpair = {}

        for i, item in enumerate(sorted_pred_result):
            # Save entpair label and result for later calculating F1
            idtf = item['entpair'][0] + '#' + item['entpair'][1]
            if idtf not in entpair:
                entpair[idtf] = {
                    'label': np.zeros((len(self.rel2id)), dtype=np.int),
                    'pred': np.zeros((len(self.rel2id)), dtype=np.int),
                    'score': np.zeros((len(self.rel2id)), dtype=np.float)
                }
            if (item['entpair'][0], item['entpair'][1], item['relation']) in self.facts:
                correct += 1
                entpair[idtf]['label'][self.rel2id[item['relation']]] = 1
            if item['score'] >= threshold:
                entpair[idtf]['pred'][self.rel2id[item['relation']]] = 1
            entpair[idtf]['score'][self.rel2id[item['relation']]] = item['score']

            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
            
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec) 
        max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        best_threshold = sorted_pred_result[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()]['score']
        mean_prec = np_prec.mean()

        label_vec = []
        pred_result_vec = []
        score_vec = []
        for ep in entpair:
            label_vec.append(entpair[ep]['label'])
            pred_result_vec.append(entpair[ep]['pred'])
            score_vec.append(entpair[ep]['score'])
        label_vec = np.stack(label_vec, 0)
        pred_result_vec = np.stack(pred_result_vec, 0)
        score_vec = np.stack(score_vec, 0)

        micro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')
        micro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')
        micro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')

        macro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        macro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        
        pred_result_vec = score_vec >= best_threshold
        max_macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        max_micro_f1_each_relation = {}
        for rel in self.rel2id:
            if rel != 'NA':
                max_micro_f1_each_relation[rel] = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=[self.rel2id[rel]], average='micro')

        return {'np_prec': np_prec, 'np_rec': np_rec, 'max_micro_f1': max_micro_f1, 'max_macro_f1': max_macro_f1, 'auc': auc, 'p@100': np_prec[99], 'p@200': np_prec[199], 'p@300': np_prec[299], 'avg_p300': (np_prec[99] + np_prec[199] + np_prec[299]) / 3, 'micro_f1': micro_f1, 'macro_f1': macro_f1, 'max_micro_f1_each_relation': max_micro_f1_each_relation}

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


class MultiLabelSentenceREDataset(data.Dataset):
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
    
    def eval(self, pred_score, threshold=0.5, use_name=False):
        """
        Args:
            pred_score: [sent_num, label_num]
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        assert len(self.data) == len(pred_score)
        pred_score = np.array(pred_score)
       
        # Calculate AUC
        sorted_result = []
        total = 0
        for sent_id in range(len(self.data)):
            for rel in self.rel2id:
                if rel not in ['NA', 'na', 'N/A', 'None', 'none', 'n/a', 'no_relation']:
                    sorted_result.append({'sent_id': sent_id, 'relation': rel, 'score': pred_score[sent_id][self.rel2id[rel]]})
                    if 'anno_relation_list' in self.data[sent_id]:
                        if rel in self.data[sent_id]['anno_relation_list']:
                            total += 1
                    else:
                        if rel == self.data[sent_id]['relation']:
                            total += 1

        sorted_result.sort(key=lambda x: x['score'], reverse=True)
        prec = []
        rec = []
        correct = 0
        for i, item in enumerate(sorted_result):
            if 'anno_relation_list' in self.data[item['sent_id']]:
                if item['relation'] in self.data[item['sent_id']]['anno_relation_list']:
                    correct += 1
            else:
                if item['relation'] == self.data[item['sent_id']]['relation']:
                    correct += 1 
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec) 
        max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        max_micro_f1_threshold = sorted_result[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()]['score']
        mean_prec = np_prec.mean()

        # Calculate F1
        pred_result_vec = np.zeros((len(self.data), len(self.rel2id)), dtype=np.int)
        pred_result_vec[pred_score >= threshold] = 1
        label_vec = []
        for item in self.data:
            if 'anno_relation_list' in item:
                label_vec.append(np.array(item['anno_relation_vec'], dtype=np.int))
            else:
                one_hot = np.zeros((len(self.rel2id)), dtype=np.int)
                one_hot[self.rel2id[item['relation']]] = 1
                label_vec.append(one_hot)
        label_vec = np.stack(label_vec, 0) 
        assert label_vec.shape == pred_result_vec.shape

        micro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')
        micro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')
        micro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='micro')

        macro_p = sklearn.metrics.precision_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        macro_r = sklearn.metrics.recall_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')
        macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec, labels=list(range(1, len(self.rel2id))), average='macro')

        acc = (label_vec == pred_result_vec).mean()

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1, 'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1, 'np_prec': np_prec, 'np_rec': np_rec, 'max_micro_f1': max_micro_f1, 'max_micro_f1_threshold': max_micro_f1_threshold, 'auc': auc, 'p@100': np_prec[99], 'p@200': np_prec[199], 'p@300': np_prec[299]}
        logging.info('Evaluation result: {}.'.format(result))
        return result
    
def MultiLabelSentenceRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = MultiLabelSentenceREDataset(path = path, rel2id = rel2id, tokenizer = tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader


