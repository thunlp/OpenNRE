rel_set = set()

with open('train.txt', encoding='utf-8') as tf, open('train_noNA.txt', 'w', encoding='utf-8') as tfo:
    for line in tf:
        ls = line.strip().split('\t')
        rel_set.add(ls[4])
        if ls[4] != "NA":
           tfo.write(line)
print(len(rel_set))
