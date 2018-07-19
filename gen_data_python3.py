import numpy as np
import os
import json
import os

# folder of training datasets
data_path = "./origin_data/"
# files to export data
export_path = "./data/"
if not os.path.exists('./data'):
    os.mkdir('./data')
#length of sentence
fixlen = 120
#max length of position embedding is 100 (-100~+100)
maxlen = 100

word2id = {}
relation2id = {}
word_size = 0
word_vec = None

def pos_embed(x):
    return max(0, min(x + maxlen, maxlen + maxlen + 1))

def find_index(x,y):
    for index, item in enumerate(y):
        if x == item:
            return index
    return -1

def init_word():
    # reading word embedding data...
    global word2id, word_size
    print('reading word embedding data...')
    f = open(data_path + 'vec.txt', "r", encoding='utf-8')
    total, size = f.readline().strip().split()[:2]
    total = (int)(total)
    word_size = (int)(size)
    vec = np.ones((total, word_size), dtype = np.float32)
    for i in range(total):
        content = f.readline().strip().split()
        word2id[content[0]] = len(word2id)
        for j in range(word_size):
            vec[i][j] = (float)(content[j+1])
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    global word_vec
    word_vec = vec

def init_relation():
    # reading relation ids...
    global relation2id
    print('reading relation ids...')
    f = open(data_path + "relation2id.txt","r", encoding='utf-8')
 
    total = (int)(f.readline().strip())
    for i in range(total):
        content = f.readline().strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

def sort_files(name):
    hash = {}
    f = open(data_path + name + '.txt','r', encoding='utf-8')

    s = 0
    while True:
        content = f.readline()
        if content == '':
            break
        s = s + 1
        origin_data = content
        content = content.strip().split()
        en1_id = content[0]
        en2_id = content[1]
        rel_name = content[4]
        if rel_name in relation2id:
            relation = relation2id[rel_name]
        else:
            relation = relation2id['NA']
        id = str(en1_id)+"#"+str(en2_id)+"#"+str(relation)
        if not id in hash:
            hash[id] = []
        hash[id].append(origin_data)
    f.close()
    f = open(data_path + name + "_sort.txt", "w", encoding='utf-8')

    f.write("%d\n"%(s))
    for i in hash:
        for j in hash[i]:
            f.write(j)
    f.close()

def init_train_files(name):
    print('reading ' + name +' data...')
    f = open(data_path + name + '.txt','r', encoding='utf-8')

    total = (int)(f.readline().strip())
    sen_word = np.zeros((total, fixlen), dtype = np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype = np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype = np.int32)
    sen_mask = np.zeros((total, fixlen), dtype = np.int32)
    sen_len = np.zeros((total), dtype = np.int32)
    sen_label = np.zeros((total), dtype = np.int32)
    instance_scope = []
    instance_triple = []
    for s in range(total):
        content = f.readline().strip().split()
        sentence = content[5:-1]
        en1_id = content[0]
        en2_id = content[1]
        en1_name = content[2]
        en2_name = content[3]
        rel_name = content[4]
        if rel_name in relation2id:
            relation = relation2id[rel_name]
        else:
            relation = relation2id['NA']
        en1pos = 0
        en2pos = 0
        for i in range(len(sentence)):
            if sentence[i] == en1_name:
                en1pos = i
            if sentence[i] == en2_name:
                en2pos = i
        en_first = min(en1pos,en2pos)
        en_second = en1pos + en2pos - en_first
        for i in range(fixlen):
            sen_word[s][i] = word2id['BLANK']
            sen_pos1[s][i] = pos_embed(i - en1pos)
            sen_pos2[s][i] = pos_embed(i - en2pos)
            if i >= len(sentence):
                sen_mask[s][i] = 0
            elif i - en_first<=0:
                sen_mask[s][i] = 1
            elif i - en_second<=0:
                sen_mask[s][i] = 2
            else:
                sen_mask[s][i] = 3
        for i, word in enumerate(sentence):
            if i >= fixlen:
                break
            elif not word in word2id:
                sen_word[s][i] = word2id['UNK']
            else:
                sen_word[s][i] = word2id[word]
        sen_len[s] = min(fixlen, len(sentence))
        sen_label[s] = relation
        tup = (en1_id,en2_id,relation)
        if instance_triple == [] or instance_triple[len(instance_triple) - 1] != tup:
            instance_triple.append(tup)
            instance_scope.append([s,s])
        instance_scope[len(instance_triple) - 1][1] = s
        if (s+1) % 100 == 0:
            print(s)
    return np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask
    

def init_test_files(name):
    print('reading ' + name +' data...')
    f = open(data_path + name + '.txt','r', encoding='utf-8')

    total = (int)(f.readline().strip())
    sen_word = np.zeros((total, fixlen), dtype = np.int32)
    sen_pos1 = np.zeros((total, fixlen), dtype = np.int32)
    sen_pos2 = np.zeros((total, fixlen), dtype = np.int32)
    sen_mask = np.zeros((total, fixlen), dtype = np.int32)
    sen_len = np.zeros((total), dtype = np.int32)
    sen_label = np.zeros((total), dtype = np.int32)
    instance_scope = []
    instance_triple = []
    instance_triple_with_NA = []
    instance_entity = []
    instance_entity_no_bag = []

    ss = [] 
    for s in range(total):
        content = f.readline().strip().split()
        sentence = content[5:-1]
        en1_id = content[0]
        en2_id = content[1]
        en1_name = content[2]
        en2_name = content[3]
        rel_name = content[4]

        ss.append((en1_id + '\t' + en2_id + '\t' + rel_name, sentence, en1_id, en2_id, en1_name, en2_name, rel_name))
    
    ss = sorted(ss, key=lambda x:x[0])
        
    for s in range(total):
        unique_id, sentence, en1_id, en2_id, en1_name, en2_name, rel_name = ss[s]
        if rel_name in relation2id:
            relation = relation2id[rel_name]
        else:
            relation = relation2id['NA']
        en1pos = 0
        en2pos = 0
        for i in range(len(sentence)):
            if sentence[i] == en1_name:
                en1pos = i
            if sentence[i] == en2_name:
                en2pos = i
        en_first = min(en1pos,en2pos)
        en_second = en1pos + en2pos - en_first
        for i in range(fixlen):
            sen_word[s][i] = word2id['BLANK']
            sen_pos1[s][i] = pos_embed(i - en1pos)
            sen_pos2[s][i] = pos_embed(i - en2pos)
            if i >= len(sentence):
                sen_mask[s][i] = 0
            elif i - en_first<=0:
                sen_mask[s][i] = 1
            elif i - en_second<=0:
                sen_mask[s][i] = 2
            else:
                sen_mask[s][i] = 3
        for i, word in enumerate(sentence):
            if i >= fixlen:
                break
            elif not word in word2id:
                sen_word[s][i] = word2id['UNK']
            else:
                sen_word[s][i] = word2id[word]
        sen_len[s] = min(fixlen, len(sentence))
        sen_label[s] = relation
        tup = (en1_id, en2_id, int(relation))
        instance_entity_no_bag.append(tup[:2])
        if instance_triple_with_NA == [] or instance_triple_with_NA[len(instance_triple_with_NA) - 1] != tup:
            if instance_triple_with_NA == [] or instance_triple_with_NA[len(instance_triple_with_NA) - 1][:2] != tup[:2]:
                instance_scope.append([s,s])
                instance_entity.append(tup[:2])
            instance_triple_with_NA.append(tup)
            if tup[2] != 0:
                instance_triple.append(tup)
        instance_scope[len(instance_scope) - 1][1] = s
        if (s+1) % 100 == 0:
            print(s)
    return np.array(instance_entity), np.array(instance_entity_no_bag), np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask
 

init_word()
init_relation()
np.save(export_path+'vec', word_vec)
f = open(export_path+'config', "w", encoding='utf-8')

f.write(json.dumps({"word2id":word2id,"relation2id":relation2id,"word_size":word_size, "fixlen":fixlen, "maxlen":maxlen}))
f.close()

sort_files("train")
sort_files("test")

instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask = init_train_files("train_sort")
np.save(export_path+'train_instance_triple', instance_triple)
np.save(export_path+'train_instance_scope', instance_scope)
np.save(export_path+'train_len', train_len)
np.save(export_path+'train_label', train_label)
np.save(export_path+'train_word', train_word)
np.save(export_path+'train_pos1', train_pos1)
np.save(export_path+'train_pos2', train_pos2)
np.save(export_path+'train_mask', train_mask)

instance_entity, instance_entity_no_bag, instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask = init_test_files("test_sort")
np.save(export_path+'test_instance_entity', instance_entity)
np.save(export_path+'test_instance_entity_no_bag', instance_entity_no_bag)
np.save(export_path+'test_instance_triple', instance_triple)
np.save(export_path+'test_instance_scope', instance_scope)
np.save(export_path+'test_len', test_len)
np.save(export_path+'test_label', test_label)
np.save(export_path+'test_word', test_word)
np.save(export_path+'test_pos1', test_pos1)
np.save(export_path+'test_pos2', test_pos2)
np.save(export_path+'test_mask', test_mask)

