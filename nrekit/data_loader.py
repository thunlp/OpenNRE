import json
import os
import multiprocessing
import numpy as np
import random

class file_data_loader:
    def __next__(self):
        raise NotImplementedError
    
    def next(self):
        return self.__next__()
    
class json_file_data_loader(file_data_loader):
    MODE_INSTANCE = 0      # One batch contains batch_size instances.
    MODE_ENTPAIR_BAG = 1   # One batch contains batch_size bags, instances in which have the same entity pair (usually for testing).
    MODE_RELFACT_BAG = 2   # One batch contains batch size bags, instances in which have the same relation fact. (usually for training).

    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        rel_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_rel.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        relfact2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_relfact2scope.json')
        rel2id_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2id.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(rel_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(entpair2scope_file_name) or \
           not os.path.exists(relfact2scope_file_name) or \
           not os.path.exists(rel2id_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_rel = np.load(rel_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.entpair2scope = json.load(open(entpair2scope_file_name))
        self.relfact2scope = json.load(open(relfact2scope_file_name))
        self.rel2id = json.load(open(rel2id_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, mode, shuffle=True, max_length=120, case_sensitive=False, reprocess=False, batch_size=160):
        '''
        file_name: Json file storing the data in the following format
            [
                {
                    'sentence': 'Bill Gates is the founder of Microsoft .',
                    'head': {'word': 'Bill Gates', ...(other information)},
                    'tail': {'word': 'Microsoft', ...(other information)},
                    'relation': 'founder'
                },
                ...
            ]
        word_vec_file_name: Json file storing word vectors in the following format
            {
                'the': [0.418, 0.24968, ...],
                ',': [0.013441, 0.23682, ...],
                ...
            }
        mode: Specify how to get a batch of data. See MODE_* constants for details.
        shuffle: Whether to shuffle the data, default as True. You should use shuffle when training.
        max_length: The length that all the sentences need to be extend to, default as 120.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        batch_size: The size of each batch, default as 160.
        '''

        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size

        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            self.ori_word_vec = None
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for i in range(len(self.ori_data)):
                    self.ori_data[i]['sentence'] = self.ori_data[i]['sentence'].lower()
                    self.ori_data[i]['head']['word'] = self.ori_data[i]['head']['word'].lower()
                    self.ori_data[i]['tail']['word'] = self.ori_data[i]['tail']['word'].lower()
                print("Finish eliminating")

            # Sort data by entities and relations
            print("Sort data...")
            def compare_by_entities_and_relations(a, b):
                a_key = a['head']['word'] + '#' + a['tail']['word'] + '#' + a['relation']
                b_key = b['head']['word'] + '#' + b['tail']['word'] + '#' + b['relation']
                if a_key > b_key:
                    return 1
                elif a_key == b_key:
                    return 0
                else:
                    return -1
            self.ori_data.sort(cmp=compare_by_entities_and_relations)
            print("Finish sorting")
       
            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec) + 2 # +BLANK and UNK
            self.word_vec_dim = len(self.ori_word_vec.values()[0])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            cur_id = 2
            for word in self.ori_word_vec.keys():
                self.word2id[word] = cur_id
                self.word_vec_mat[cur_id, :] = self.ori_word_vec[word]
                cur_id += 1 
            self.word_vec_mat[1, :] = self.word_vec_mat[2:].mean(0)
            self.word2id['BLANK'] = 0
            self.word2id['UNK'] = 1
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = len(self.ori_data)
            self.entpair2scope = {} # (head, tail) -> scope
            self.relfact2scope = {} # (head, tail, relation) -> scope
            self.rel2id = {} # relation -> id
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32) # [start_pos, end_pos], left closed right open
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_rel = np.zeros((self.instance_tot), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            last_entpair = ''
            last_entpair_pos = -1
            last_relfact = ''
            last_relfact_pos = -1
            for i in range(self.instance_tot):
                ins = self.ori_data[i]
                if not ins['relation'] in self.rel2id:
                    self.rel2id[ins['relation']] = len(self.rel2id)
                self.data_rel[i] = self.rel2id[ins['relation']]
                sentence = ' '.join(ins['sentence'].split()) # delete extra spaces
                head = ins['head']['word']
                tail = ins['tail']['word']
                cur_entpair = head + '#' + tail
                cur_relfact = head + '#' + tail + '#' + ins['relation']
                if cur_entpair != last_entpair:
                    if last_entpair != '':
                        self.entpair2scope[last_entpair] = [last_entpair_pos, i] # left closed right open
                    last_entpair = cur_entpair
                    last_entpair_pos = i
                if cur_relfact != last_relfact:
                    if last_relfact != '':
                        self.relfact2scope[last_relfact] = [last_relfact_pos, i]
                    last_relfact = cur_relfact
                    last_relfact_pos = i
                p1 = sentence.find(' ' + head + ' ') + 1
                p2 = sentence.find(' ' + tail + ' ') + 1
                if p1 == -1:
                    if sentence[:len(head)] == head:
                        p1 = 0
                    elif sentence[-len(head):] == head:
                        p1 = len(sentence) - len(head)
                if p2 == -1:
                    if sentence[:len(tail)] == tail:
                        p2 = 0
                    elif sentence[-len(tail):] == tail:
                        p2 = len(sentence) - len(tail)
                if p1 == -1 or p2 == -1:
                    raise Exception("[ERROR] Sentence doesn't contain the entity, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))
                words = sentence.split()
                cur_ref_data_word = self.data_word[i]         
                cur_pos = 0
                pos1 = -1
                pos2 = -1
                for j, word in enumerate(words):
                    if j < max_length:
                        if word in self.word2id:
                            cur_ref_data_word[j] = self.word2id[word]
                        else:
                            cur_ref_data_word[j] = 1
                    if cur_pos == p1:
                        pos1 = j
                        p1 = -1
                    if cur_pos == p2:
                        pos2 = j
                        p2 = -1
                    cur_pos += len(word) + 1
                self.data_length[i] = len(words)
                if len(words) > max_length:
                    self.data_length[i] = max_length
                if pos1 == -1 or pos2 == -1:
                    raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))
                for j in range(max_length):
                    if cur_ref_data_word[j] == 0:
                        break
                    self.data_pos1[i][j] = j - pos1 + max_length
                    self.data_pos2[i][j] = j - pos2 + max_length
            if last_entpair != '':
                self.entpair2scope[last_entpair] = [last_entpair_pos, self.instance_tot] # left closed right open
            if last_relfact != '':
                self.relfact2scope[last_relfact] = [last_relfact_pos, self.instance_tot]

            print("Finish pre-processing")     

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_rel.npy'), self.data_rel)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.entpair2scope, open(os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json'), 'w'))
            json.dump(self.relfact2scope, open(os.path.join(processed_data_dir, name_prefix + '_relfact2scope.json'), 'w'))
            json.dump(self.rel2id, open(os.path.join(processed_data_dir, name_prefix + '_rel2id.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")

        # Prepare for idx
        self.instance_tot = self.data_word.shape[0]
        self.entpair_tot = len(self.entpair2scope)
        self.relfact_tot = len(self.relfact2scope)
        self.rel_tot = len(self.rel2id)

        if self.mode == self.MODE_INSTANCE:
            self.order = list(range(self.instance_tot))
        elif self.mode == self.MODE_ENTPAIR_BAG:
            self.order = list(range(self.entpair_tot))
            self.scope = self.entpair2scope.values()
        elif self.mode == self.MODE_RELFACT_BAG:
            self.order = list(range(self.relfact_tot))
            self.scope = self.relfact2scope.values()
        else:
            raise Exception("[ERROR] Invalid mode")
        self.idx = 0

        if self.shuffle:
            random.shuffle(self.order) 

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.order):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.order) 
            raise StopIteration

        batch_data = {}

        if self.mode == self.MODE_INSTANCE:
            idx0 = self.idx
            idx1 = self.idx + self.batch_size
            if idx1 > len(self.order):
                idx1 = len(self.order)
            self.idx = idx1
            batch_data['word'] = self.data_word[idx0:idx1]
            batch_data['pos1'] = self.data_pos1[idx0:idx1]
            batch_data['pos2'] = self.data_pos2[idx0:idx1]
            batch_data['rel'] = self.data_rel[idx0:idx1]
            batch_data['length'] = self.data_length[idx0:idx1]
            batch_data['scope'] = np.stack([list(range(idx1 - idx0)), list(range(1, idx1 - idx0 + 1))], axis=1)
        elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
            idx0 = self.idx
            idx1 = self.idx + self.batch_size
            if idx1 > len(self.order):
                idx1 = len(self.order)
            self.idx = idx1
            _word = []
            _pos1 = []
            _pos2 = []
            _rel = []
            _length = []
            _scope = []
            cur_pos = 0
            for i in range(idx0, idx1):
                _word.append(self.data_word[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos1.append(self.data_pos1[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos2.append(self.data_pos2[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _rel.append(self.data_rel[self.scope[self.order[i]][0]])
                _length.append(self.data_length[self.scope[self.order[i]][0]])

                bag_size = self.scope[self.order[i]][1] - self.scope[self.order[i]][0]
                _scope.append([cur_pos, cur_pos + bag_size])
                cur_pos = cur_pos + bag_size
            batch_data['word'] = np.concatenate(_word)
            batch_data['pos1'] = np.concatenate(_pos1)
            batch_data['pos2'] = np.concatenate(_pos2)
            batch_data['rel'] = np.stack(_rel)
            batch_data['length'] = np.stack(_length)
            batch_data['scope'] = np.stack(_scope)

        return batch_data
