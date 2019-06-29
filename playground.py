import nrekit

#fewshot_model = nrekit.fewshot_re_model(name='proto_cnn')
fewshot_model = nrekit.fewshot_re_model(name='bert_pair')
support = [
    {
        'token': ['Bill', 'Gates', 'is', 'the', 'founder', 'of', 'Microsoft', '.'],
        'h': {'name': 'Bill Gates', 'pos': [0, 2]},
        't': {'name': 'Microsoft', 'pos': [6, 7]},
        'relation': 'founder'
    },
    {
        'token': ['Linda', 'was', 'an', 'employer', 'of', 'Microsoft', '.'],
        'h': {'name': 'Linda', 'pos': [0, 1]},
        't': {'name': 'Microsoft', 'pos': [5, 6]},
        'relation': 'employer'
    }
]
query = [
    {
        'token': ['Apple', 'was', 'founded', 'by', 'Steve', 'Jobs', '.'],
        'h': {'name': 'Steve Jobs', 'pos': [4, 6]},
        't': {'name': 'Apple', 'pos': [0, 1]}
    },
    {
        'token': ['Eric', 'is', 'an', 'employer', 'in', 'Apple', '.'],
        'h': {'name': 'Eric', 'pos': [0, 1]},
        't': {'name': 'Apple', 'pos': [5, 6]}
    }
]

result = fewshot_model.infer(support, query)
print(result)
