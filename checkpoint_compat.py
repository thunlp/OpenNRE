import tensorflow as tf
from pprint import pprint


def get_compat_dict(filepath="./model/ATT_GRU_model-10900"):
    compat_dict = {"MultiRNNCell": "multi_rnn_cell",
                   "Cell0": "cell_0",
                   "GRUCell": "gru_cell",
                   "Candidate": "candidate",
                   "Gates": "gates",
                   "Bias": "biases",
                   "Matrix": "weights",
                   "Adam": "adam",
                   "Adam_1": "adam_1"}

    ckpt_reader = tf.train.NewCheckpointReader(filepath)
    ckpt_vars = ckpt_reader.get_variable_to_shape_map()
    old_to_new = {}
    for key in ckpt_vars:
        tokens = key.split('/')
        new_tokens = []
        for token in tokens:
            if token == 'Linear':
                continue
            elif token in compat_dict:
                new_tokens.append(compat_dict[token])
            else:
                new_tokens.append(token)
        old_to_new[key] = '/'.join(new_tokens)
    return old_to_new


def transform_name_var_dict(names_to_vars):
    old_to_new = get_compat_dict(filepath="./model/ATT_GRU_model-10900")
    print("USE LEGACY MODEL FILE")
    print("OLD name_to_vars")
    pprint(names_to_vars)
    for old in old_to_new:
        new = old_to_new[old]
        if old != new and new in names_to_vars:
            new_var = names_to_vars[new]
            names_to_vars[old] = new_var
            del names_to_vars[new]
    print("NEW name_to_vars")
    pprint(names_to_vars)
    return names_to_vars
