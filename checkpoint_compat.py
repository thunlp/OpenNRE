import tensorflow as tf
from pprint import pprint


def get_compat_dict(filepath):
    compat_dict = {"MultiRNNCell": "multi_rnn_cell",
                   "Cell0": "cell_0",
                   "GRUCell": "gru_cell",
                   "Candidate": "candidate",
                   "Gates": "gates",
                   "Linear": "linear",
                   "Bias": "biases",
                   "Matrix": "matrix",
                   "Adam": "adam",
                   "Adam_1": "adam_1"}

    ckpt_reader = tf.train.NewCheckpointReader(filepath)
    ckpt_vars = ckpt_reader.get_variable_to_shape_map()
    old_to_new = {}
    for key in ckpt_vars:
        tokens = key.split('/')
        new_tokens = []
        for token in tokens:
            if token in compat_dict:
                new_tokens.append(compat_dict[token])
            else:
                new_tokens.append(token)
        old_to_new[key] = '/'.join(new_tokens)
    return old_to_new

# sess = tf.Session()
# saver = tf.train.Saver(old_to_new)
# saver.restore(sess, filepath)
# filepath = 'model/ATT_GRU_model-10900'
