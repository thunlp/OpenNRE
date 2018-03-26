import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import utils
import tqdm
from tensorflow.contrib.tensorboard.plugins import projector
import subprocess

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')


def main(_):
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "3"
    # the path to save models
    save_path = './model/kbp/'

    print('reading wordembedding')
    wordembedding = np.load('./data/KBP/vec.npy')

    print('reading training data')
    train_y = np.load('./data/KBP/train_y.npy')
    train_word = np.load('./data/KBP/train_word.npy')
    train_pos1 = np.load('./data/KBP/train_pos1.npy')
    train_pos2 = np.load('./data/KBP/train_pos2.npy')

    none_ind = utils.get_none_id('./origin_data/KBP/relation2id.txt')
    print("None index: ", none_ind)
    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])
    print("vocab_size: ", settings.vocab_size)
    print("num_classes: ", settings.num_classes)

    best_f1 = float('-inf')
    best_recall = 0
    best_precision = 0

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.GradientDescentOptimizer(0.001)
            optimizer = tf.train.AdamOptimizer(0.001)

            # train_op=optimizer.minimize(m.total_loss,global_step=global_step)
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)

            # merged_summary = tf.summary.merge_all()
            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            # summary for embedding
            # it's not available in tf 0.11,
            # (because there is no embedding panel in 0.11's tensorboard) so I delete it =.=
            # you can try it on 0.12 or higher versions but maybe you should change some function name at first.

            # summary_embed_writer = tf.train.SummaryWriter('./model',sess.graph)
            # config = projector.ProjectorConfig()
            # embedding_conf = config.embedding.add()
            # embedding_conf.tensor_name = 'word_embedding'
            # embedding_conf.metadata_path = './data/metadata.tsv'
            # projector.visualize_embeddings(summary_embed_writer, config)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                accuracy = np.reshape(np.array(accuracy), big_num)
                summary_writer.add_summary(summary, step)
                return step, loss, accuracy

            # training process
            for one_epoch in range(settings.num_epochs):
                print("Starting Epoch: ", one_epoch)
                epoch_loss = 0
                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)

                all_prob = []
                all_true = []
                all_accuracy = []
                for i in tqdm.tqdm(range(int(len(temp_order) / float(settings.big_num)))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)

                    step, loss, accuracy = train_step(temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)
                    epoch_loss += loss
                    all_accuracy.append(accuracy)

                    all_true.append(temp_y)
                accu = np.mean(all_accuracy)
                print("Epoch finished, loss:, ", epoch_loss, "accu: ", accu)

                # all_prob = np.concatenate(all_prob, axis=0)
                # all_true = np.concatenate(all_true, axis=0)
                #
                # all_pred_inds = utils.calcInd(all_prob)
                # entropy = utils.calcEntropy(all_prob)
                # all_true_inds = np.argmax(all_true, 1)
                # f1score, recall, precision, meanBestF1 = utils.CrossValidation(all_pred_inds, entropy,
                #                                                                all_true_inds, none_ind)
                # print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f)' %
                #       (f1score,
                #        recall,
                #        precision,
                #        meanBestF1))
                print('saving model')
                current_step = tf.train.global_step(sess, global_step)
                path = saver.save(sess, save_path + 'ATT_GRU_model', global_step=current_step)
                print(path)
                print("start testing")
                subprocess.run(['python3', 'test_GRU.py', str(current_step)], env=my_env)


if __name__ == "__main__":
    tf.app.run()
