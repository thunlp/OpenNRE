import json
from pycorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))


def get_stanford_annotations(text, annotators='tokenize,ssplit'):
    output = nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.isOneSentence": "true",
        'annotators': annotators,
    })
    return output


def tokenize(text):
    annotations = get_stanford_annotations(text)
    annotations = json.loads(annotations, encoding="utf-8", strict=False)
    tokens = annotations['sentences'][0]['tokens']
    return [token['word'] for token in tokens]


def process_file(in_file, out_file, rel_types, training=True):
    for line in in_file:
        data = json.loads(line.strip())
        sent_tokens_lower = [t.lower() for t in data['tokens']]
        rms = data['relationMentions']
        for rm in rms:
            relations = rm['labels']
            for r in relations:
                if training:
                    if r not in rel_types:
                        rel_types[r] = len(rel_types)
                em1_start = rm['em1Start']
                em1_end = rm['em1End']
                em2_start = rm['em2Start']
                em2_end = rm['em2End']
                new_sent_str = ""
                for idx, token in enumerate(sent_tokens_lower):
                    if em1_start <= idx < em1_end - 1 or em2_start <= idx < em2_end - 1:
                        new_sent_str += token + '_'
                    else:
                        new_sent_str += token + ' '
                em1 = '_'.join(sent_tokens_lower[em1_start:em1_end])
                em2 = '_'.join(sent_tokens_lower[em2_start:em2_end])
                out_file.write('_\t_\t' + em1 + '\t' + em2 + '\t' + r + '\t' + new_sent_str + '\n')
    return rel_types


if __name__ == '__main__':
    dataset = 'NYT'
    with open("origin_data/" + dataset + "/train_new.json", encoding='utf-8') as train_file, \
            open("origin_data/" + dataset + "/test_new.json", encoding='utf-8') as test_file, \
            open("origin_data/" + dataset + "/relation2id.txt", 'w', encoding='utf-8') as relation_file, \
            open("origin_data/" + dataset + "/train.txt", 'w', encoding='utf-8') as train_out_file, \
            open("origin_data/" + dataset + "/test.txt", 'w', encoding='utf-8') as test_out_file:
        rel_types = {}
        rel_types = process_file(train_file, train_out_file, rel_types, training=True)
        process_file(test_file, test_out_file, rel_types, training=False)
        for relation in rel_types:
            relation_file.write(relation + ' ' + str(rel_types[relation]) + '\n')