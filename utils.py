import random
import numpy as np


def get_none_id(type_filename):
    with open(type_filename, encoding='utf-8') as type_file:
        for line in type_file:
            ls = line.strip().split()
            if ls[0] == "None":
                return int(ls[1])


def evaluate_rm_neg(prediction, ground_truth, none_label_index):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    # print '[None] label index:', none_label_index

    pos_pred = 0.0
    pos_gt = 0.0
    true_pos = 0.0
    for i in range(len(ground_truth)):
        if ground_truth[i] != none_label_index:
            pos_gt += 1.0

    for i in range(len(prediction)):
        if prediction[i] != none_label_index:
            # classified as pos example (Is-A-Relation)
            pos_pred += 1.0
            if prediction[i] == ground_truth[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1

