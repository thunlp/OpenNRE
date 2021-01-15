import json
import ipdb

file_path = 'benchmark/tacred/dev.json'
write_file = 'benchmark/tacred/tacred_val.txt'
rel2id_file = 'benchmark/tacred/rel2id.txt'
relations = []
with open(file_path, 'r') as f1, open(write_file, 'w') as f2:
    jss = json.load(f1)
    for js in jss:
        res = {}
        res['token'] = js['token']
        res['h'] = {}
        res['h']['pos'] = [js['subj_start'], js['subj_end']]
        res['t'] = {}
        res['t']['pos'] = [js['obj_start'], js['obj_end']]
        res['relation'] = js['relation']
        relations.append(res['relation'])
        f2.write(json.dumps(res))
        f2.write('\n')

# with open(rel2id_file, 'w') as f:
#     rel2id = {}
#     relations = list(set(relations))
#     ipdb.set_trace()
#     for i in range(len(relations)):
#         rel2id[relations[i]] = i
#
#     f.write(json.dumps(rel2id))