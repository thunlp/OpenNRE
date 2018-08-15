import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import datetime
import sys
from network.embedding import Embedding
from network.encoder import Encoder
from network.selector import Selector
from network.classifier import Classifier
import os
import sklearn.metrics

import time
import math

FLAGS = tf.app.flags.FLAGS

class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0

class Framework(object):

    def __init__(self, is_training, use_bag=True):
        self.use_bag = use_bag
        # Place Holder
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_word')
        #self.word_vec = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.word_size], name='word_vec')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_pos2')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='input_length')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='input_mask')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
        self.label_for_select = tf.placeholder(dtype=tf.int32, shape=[None], name='label_for_select')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size + 1], name='scope')    
        self.weights = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size])

        self.data_word_vec = np.load(os.path.join(FLAGS.export_path, 'vec.npy'))

        # Network
        self.embedding = Embedding(is_training, self.data_word_vec, self.word, self.pos1, self.pos2)
        self.encoder = Encoder(is_training, FLAGS.drop_prob)
        self.selector = Selector(FLAGS.num_classes, is_training, FLAGS.drop_prob)
        self.classifier = Classifier(is_training, self.label, self.weights)

        # Metrics 
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.step = 0
        
        # Session
        self.sess = None

    def load_train_data(self):
        print('reading training data...')
        #self.data_word_vec = np.load(os.path.join(FLAGS.export_path, 'vec.npy'))
        self.data_instance_triple = np.load(os.path.join(FLAGS.export_path, 'train_instance_triple.npy'))
        self.data_instance_scope = np.load(os.path.join(FLAGS.export_path, 'train_instance_scope.npy'))
        self.data_train_length = np.load(os.path.join(FLAGS.export_path, 'train_len.npy'))
        self.data_train_label = np.load(os.path.join(FLAGS.export_path, 'train_label.npy'))
        self.data_train_word = np.load(os.path.join(FLAGS.export_path, 'train_word.npy'))
        self.data_train_pos1 = np.load(os.path.join(FLAGS.export_path, 'train_pos1.npy'))
        self.data_train_pos2 = np.load(os.path.join(FLAGS.export_path, 'train_pos2.npy'))
        self.data_train_mask = np.load(os.path.join(FLAGS.export_path, 'train_mask.npy'))

        print('reading finished')
        print('mentions         : %d' % (len(self.data_instance_triple)))
        print('sentences        : %d' % (len(self.data_train_length)))
        print('relations        : %d' % (FLAGS.num_classes))
        print('word size        : %d' % (FLAGS.word_size))
        print('position size     : %d' % (FLAGS.pos_size))
        print('hidden size        : %d' % (FLAGS.hidden_size))

        self.reltot = {}
        for index, i in enumerate(self.data_train_label):
            if not i in self.reltot:
                self.reltot[i] = 1.0
            else:
                self.reltot[i] += 1.0
        for i in self.reltot:
            self.reltot[i] = 1 / (self.reltot[i] ** (0.05))
        print(self.reltot)

    def load_test_data(self):
        print('reading test data...')
        #self.data_word_vec = np.load(os.path.join(FLAGS.export_path, 'vec.npy'))
        self.data_instance_entity = np.load(os.path.join(FLAGS.export_path, 'test_instance_entity.npy'))
        self.data_instance_entity_no_bag = np.load(os.path.join(FLAGS.export_path, 'test_instance_entity_no_bag.npy'))
        instance_triple = np.load(os.path.join(FLAGS.export_path, 'test_instance_triple.npy'))
        self.data_instance_triple = {}
        for item in instance_triple:
            self.data_instance_triple[(item[0], item[1], int(item[2]))] = 0
        self.data_instance_scope = np.load(os.path.join(FLAGS.export_path, 'test_instance_scope.npy'))
        self.data_test_length = np.load(os.path.join(FLAGS.export_path, 'test_len.npy'))
        self.data_test_label = np.load(os.path.join(FLAGS.export_path, 'test_label.npy'))
        self.data_test_word = np.load(os.path.join(FLAGS.export_path, 'test_word.npy'))
        self.data_test_pos1 = np.load(os.path.join(FLAGS.export_path, 'test_pos1.npy'))
        self.data_test_pos2 = np.load(os.path.join(FLAGS.export_path, 'test_pos2.npy'))
        self.data_test_mask = np.load(os.path.join(FLAGS.export_path, 'test_mask.npy'))

        print('reading finished')
        print('mentions         : %d' % (len(self.data_instance_triple)))
        print('sentences        : %d' % (len(self.data_test_length)))
        print('relations        : %d' % (FLAGS.num_classes))
        print('word size        : %d' % (FLAGS.word_size))
        print('position size     : %d' % (FLAGS.pos_size))
        print('hidden size        : %d' % (FLAGS.hidden_size))

    def init_train_model(self, loss, output, optimizer=tf.train.GradientDescentOptimizer):
        print('initializing training model...')

        # Loss and output
        self.loss = loss
        self.output = output

        # Optimizer
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.summary.scalar('learning_rate', FLAGS.learning_rate)
        self.optimizer = optimizer(FLAGS.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(loss)
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        # Summary
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, self.sess.graph)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=None)
        if FLAGS.pretrain_model == "None":
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, FLAGS.pretrain_model)

        print('initializing finished')

    def init_test_model(self, output):
        print('initializing test model...')
        self.output = output
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=None)
        print('initializing finished')

    def train_one_step(self, index, scope, weights, label, label_for_select, result_needed=[]):
        #print self.data_train_word[index, :].shape
        #print 'limit bag size < 1000'
        #if self.data_train_word[index, :].shape[0] > 500:
        #    return [-1]
        feed_dict = {
            self.word: self.data_train_word[index, :],
            #self.word_vec: self.data_word_vec,
            self.pos1: self.data_train_pos1[index, :],
            self.pos2: self.data_train_pos2[index, :],
            self.mask: self.data_train_mask[index, :],
            self.length: self.data_train_length[index],
            self.label: label,
            self.label_for_select: label_for_select, 
            self.scope: np.array(scope),
            self.weights: weights
        }
        result = self.sess.run([self.train_op, self.global_step, self.output] + result_needed, feed_dict)
        self.step = result[1]
        _output = result[2]
        result = result[3:]

        # Training accuracy
        for i, prediction in enumerate(_output):
            if label[i] == 0:
                self.acc_NA.add(prediction == label[i])
            else:
                self.acc_not_NA.add(prediction == label[i])
            self.acc_total.add(prediction == label[i])

        return result

    def test_one_step(self, index, scope, label, result_needed=[]):
        feed_dict = {
            self.word: self.data_test_word[index, :],
            #self.word_vec: self.data_word_vec,
            self.pos1: self.data_test_pos1[index, :],
            self.pos2: self.data_test_pos2[index, :],
            self.mask: self.data_test_mask[index, :],
            self.length: self.data_test_length[index],
            self.label: label,
            self.label_for_select: self.data_test_label[index],
            self.scope: np.array(scope),
        }
        result = self.sess.run([self.output] + result_needed, feed_dict)
        if self.use_bag:
            self.test_output = result[0]
        else:
            tmp_output = result[0]
            self.test_output = []
            for i in range(FLAGS.batch_size):
                self.test_output.append(np.mean(tmp_output[scope[i]:scope[i + 1]], axis=0))
        result = result[1:]

        return result
    
    def train(self, one_step=train_one_step):
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.mkdir(FLAGS.checkpoint_dir)
        if self.use_bag:
            train_order = list(range(len(self.data_instance_triple)))
        else:
            train_order = list(range(len(self.data_train_word)))
        for epoch in range(FLAGS.max_epoch):
            print('epoch ' + str(epoch) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            np.random.shuffle(train_order)
            for i in range(int(len(train_order) / float(FLAGS.batch_size))):
                if self.use_bag:
                    input_scope = np.take(self.data_instance_scope, train_order[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size], axis=0)
                    index = []
                    scope = [0]
                    weights = []
                    label = []
                    for num in input_scope:
                        index = index + list(range(num[0], num[1] + 1))
                        label.append(self.data_train_label[num[0]])
                        scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                        weights.append(self.reltot[self.data_train_label[num[0]]])
                    
                    loss = one_step(self, index, scope, weights, label, self.data_train_label[index], [self.loss])
                else:
                    index = list(range(i * FLAGS.batch_size, (i + 1) * FLAGS.batch_size))
                    weights = []
                    for j in index:
                        weights.append(self.reltot[self.data_train_label[j]])
                    loss = one_step(self, index, index + [0], weights, self.data_train_label[index], self.data_train_label[index], [self.loss])

                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("epoch %d step %d | loss : %f, NA accuracy: %f, not NA accuracy: %f, total accuracy %f" % (epoch, i, loss[0], self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()) + '\r')
                sys.stdout.flush()

            if (epoch + 1) % FLAGS.save_epoch == 0:
                print('epoch ' + str(epoch + 1) + ' has finished')
                print('saving model...')
                path = self.saver.save(self.sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=epoch)
                print('have saved model to ' + path)

    def test(self, one_step=test_one_step):
        epoch_range = eval(FLAGS.epoch_range)
        epoch_range = range(epoch_range[0], epoch_range[1])
        save_x = None
        save_y = None
        best_auc = 0
        best_prec_mean = 0
        best_epoch = 0
        print('test ' + FLAGS.model_name)
        if FLAGS.discard_only_one:
            test_id = []
            for i in range(len(self.data_instance_scope)):
                if self.data_instance_scope[i][0] != self.data_instance_scope[i][1]:
                    test_id.append(i)
        else:
            test_id = list(range(len(self.data_instance_scope)))
        for epoch in epoch_range:
            if not os.path.exists(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name + '-' + str(epoch) + '.index')):
                continue
            print('start testing checkpoint, iteration =', epoch)
            self.saver.restore(self.sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name + '-' + str(epoch)))
            stack_output = []
            stack_label = []
            total = int(len(test_id) / FLAGS.batch_size)

            test_result = []
            total_recall = 0 
            for i in range(total):
                input_scope = np.take(self.data_instance_scope, test_id[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size], axis=0)
                index = []
                scope = [0]
                label = []
                for num in input_scope:
                    index = index + list(range(num[0], num[1] + 1))
                    label.append(self.data_test_label[num[0]])
                    scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
    
                one_step(self, index, scope, label, [])
               
                for j in range(len(self.test_output)):
                    pred = self.test_output[j]
                    entity = self.data_instance_entity[test_id[j + i * FLAGS.batch_size]]
                    for rel in range(1, len(pred)):
                        flag = int(((entity[0], entity[1], rel) in self.data_instance_triple))
                        total_recall += flag
                        test_result.append([(entity[0], entity[1], rel), flag, pred[rel]])

                if i % 100 == 0:
                    sys.stdout.write('predicting {} / {}\r'.format(i, total))
                    sys.stdout.flush()
            
            print('\nevaluating...')

            sorted_test_result = sorted(test_result, key=lambda x: x[2])
            pr_result_x = []
            pr_result_y = []
            correct = 0
            for i, item in enumerate(sorted_test_result[::-1]):
                if item[1] == 1:
                    correct += 1
                pr_result_y.append(float(correct) / (i + 1))
                pr_result_x.append(float(correct) / total_recall)

            auc = sklearn.metrics.auc(x=pr_result_x, y=pr_result_y)
            prec_mean = (pr_result_y[100] + pr_result_y[200] + pr_result_y[300]) / 3
            print('auc: {}'.format(auc))
            print('p@(100,200,300) mean: {}'.format(prec_mean))
            # if prec_mean > best_prec_mean:
            #     best_prec_mean = prec_mean
            #     best_epoch = epoch
            #     save_x = pr_result_x
            #     save_y = pr_result_y
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                save_x = pr_result_x
                save_y = pr_result_y

        if not os.path.exists(FLAGS.test_result_dir):
            os.mkdir(FLAGS.test_result_dir)
        np.save(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + '_x.npy'), save_x)
        np.save(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + '_y.npy'), save_y)
        print('best epoch:', best_epoch)
        if FLAGS.discard_only_one:
            print('USING [DISCARD ONLY ONE INSTANCE ENTITY] MODE!')

    # adversarial part
    def adversarial(self, loss, embedding):
        perturb = tf.gradients(loss, embedding)
        perturb = tf.reshape((0.01 * tf.stop_gradient(tf.nn.l2_normalize(perturb, dim=[0, 1, 2]))), [-1, FLAGS.max_length, embedding.shape[-1]])
        embedding = embedding + perturb
        return embedding
    # adversarial part ends

    # rl part
    def init_policy_agent(self, policy_agent_loss, policy_agent_output, optimizer=tf.train.GradientDescentOptimizer):
        print('initializing policy agent...')

        # Loss and output
        self.policy_agent_loss = policy_agent_loss
        self.policy_agent_output = policy_agent_output

        # Optimizer
        self.policy_agent_global_step = tf.Variable(0, name='policy_agent_global_step', trainable=False)
        tf.summary.scalar('policy_agent_learning_rate', FLAGS.learning_rate)
        self.policy_agent_optimizer = optimizer(FLAGS.learning_rate)
        self.policy_agent_grads_and_vars = self.policy_agent_optimizer.compute_gradients(policy_agent_loss)
        self.policy_agent_train_op = self.policy_agent_optimizer.apply_gradients(self.policy_agent_grads_and_vars, global_step=self.policy_agent_global_step)

        print('policy agent initializing finished')

    def train_policy_agent_one_step(self, index, label, weights, calc_acc=False):
        feed_dict = {
            self.word: self.data_train_word[index, :],
            self.pos1: self.data_train_pos1[index, :],
            self.pos2: self.data_train_pos2[index, :],
            self.mask: self.data_train_mask[index, :],
            self.length: self.data_train_length[index],
            self.label: label,
            self.weights: weights,
        }
        result = self.sess.run([self.policy_agent_train_op, self.policy_agent_global_step, self.policy_agent_output, self.policy_agent_loss] , feed_dict)

        self.policy_agent_step = result[1]
        _output = result[2]
        result = result[3:]

        # Training accuracy
        if calc_acc:
            for i, prediction in enumerate(_output):
                self.acc_total.add(prediction == label[i])
                if label[i] != 0:
                    self.acc_not_NA.add(prediction == label[i])
        return result

    def make_action(self, index):
        feed_dict = {
            self.word: self.data_train_word[index, :],
            self.pos1: self.data_train_pos1[index, :],
            self.pos2: self.data_train_pos2[index, :],
            self.mask: self.data_train_mask[index, :],
            self.length: self.data_train_length[index]
        }
        result = self.sess.run([self.policy_agent_output], feed_dict)

        return result

    def pretrain_policy_agent(self, max_epoch=1):
        train_order = list(range(len(self.data_train_word)))
        for epoch in range(max_epoch):
            print('[pretrain policy agent] ' + 'epoch ' + str(epoch) + ' starts...')
            self.acc_total.clear()
            self.acc_not_NA.clear()
            np.random.shuffle(train_order)
            for i in range(int(len(train_order) / float(FLAGS.batch_size))):
                index = range(i * FLAGS.batch_size, (i + 1) * FLAGS.batch_size)
                policy_agent_label = self.data_train_label[index] + 0
                policy_agent_label[policy_agent_label > 0] = 1
                weights = np.ones(policy_agent_label.shape, dtype=np.float32)
                loss = self.train_policy_agent_one_step(index, policy_agent_label, weights, calc_acc=True)

                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("[pretrain policy agent] epoch %d step %d | loss : %f, accuracy: %f, accuracy of 1: %f" % (epoch, i, loss[0], self.acc_total.get(), self.acc_not_NA.get()) + '\n')
                sys.stdout.flush()
            if self.acc_total.get() > 0.9:
                break

    def train_rl(self, one_step=train_one_step):
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.mkdir(FLAGS.checkpoint_dir)
        if self.use_bag:
            train_order = list(range(len(self.data_train_instance_triple)))
        else:
            train_order = list(range(len(self.data_train_word)))
        for epoch in range(FLAGS.max_epoch):
            print('epoch ' + str(epoch) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            self.step = 0
            self.policy_agent_step = 0

            np.random.shuffle(train_order)

            # update policy agent
            tot_delete = 0
            batch_count = 0
            instance_count = 0
            tot_batch = int(len(train_order) / float(FLAGS.batch_size))
            reward = 0.0
            action_result_his = np.zeros(self.data_train_label.shape, dtype=np.int32)
            for i in range(tot_batch):
                # data prepare
                input_scope = np.take(self.data_train_instance_scope, train_order[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size], axis=0)
                index = []
                scope = [0]
                weights = []
                label = []
                for num in input_scope:
                    index = index + list(range(num[0], num[1] + 1))
                    label.append(self.data_train_label[num[0]])
                    scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                    weights.append(self.reltot[self.data_train_label[num[0]]])

                # make action
                action_result = self.make_action(index)[0]
                action_result_his[index] = action_result

                # calculate reward
                batch_label = self.data_train_label[index]
                batch_delete = np.sum(np.logical_and(batch_label != 0, action_result == 0)) 
                batch_label[action_result == 0] = 0

                feed_dict = {
                    self.word: self.data_train_word[index, :],
                    self.pos1: self.data_train_pos1[index, :],
                    self.pos2: self.data_train_pos2[index, :],
                    self.mask: self.data_train_mask[index, :],
                    self.length: self.data_train_length[index],
                    self.label: label,
                    self.label_for_select: batch_label,
                    self.scope: np.array(scope),
                    self.weights: weights
                }
                batch_loss_sum = self.sess.run([self.loss], feed_dict)[0]

                reward += batch_loss_sum
                tot_delete += batch_delete
                batch_count += 1
                
                alpha = 0.1
                if batch_count == 100:
                    reward = reward / float(batch_count)
                    average_loss = reward
                    reward = - math.log(1 - math.e ** (-reward))
                    sys.stdout.write('tot delete : %f | reward : %f | average loss : %f' % (tot_delete, reward, average_loss))
                    sys.stdout.flush()
                    for j in range(i - batch_count + 1, i + 1):
                        input_scope = np.take(self.data_train_instance_scope, train_order[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size], axis=0)
                        index = []
                        for num in input_scope:
                            index = index + list(range(num[0], num[1] + 1))
                        batch_result = action_result_his[index]
                        weights = np.ones(batch_result.shape, dtype=np.float32)
                        weights *= reward
                        weights *= alpha
                        loss = self.train_policy_agent_one_step(index, batch_result, weights)

                    batch_count = 0
                    reward = 0
                    tot_delete = 0

            for i in range(int(len(train_order) / float(FLAGS.batch_size))):
                if self.use_bag:
                    input_scope = np.take(self.data_train_instance_scope, train_order[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size], axis=0)
                    index = []
                    scope = [0]
                    weights = []
                    label = []
                    for num in input_scope:
                        index = index + list(range(num[0], num[1] + 1))
                        label.append(self.data_train_label[num[0]])
                        scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
                        weights.append(self.reltot[self.data_train_label[num[0]]])

                    # make action
                    action_result = self.make_action(index) 
    
                    # calculate reward
                    batch_label = self.data_train_label[index]
                    batch_delete = np.sum(np.logical_and(batch_label != 0, action_result == 0)) 
                    batch_label[action_result == 0] = 0
    
                    loss = one_step(self, index, scope, weights, label, batch_label, [self.loss])
                else:
                    index = range(i * FLAGS.batch_size, (i + 1) * FLAGS.batch_size)
                    weights = []
                    for i in index:
                        weights.append(self.reltot[self.data_train_label[i]])
                    loss = one_step(self, index, index + [0], weights, self.data_train_label[index], [self.loss])

                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("epoch %d step %d | loss : %f, NA accuracy: %f, not NA accuracy: %f, total accuracy %f" % (epoch, i, loss[0], self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()) + '\r')
                sys.stdout.flush()

            if (epoch + 1) % FLAGS.save_epoch == 0:
                print('epoch ' + str(epoch + 1) + ' has finished')
                print('saving model...')
                path = self.saver.save(self.sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=epoch)
                print('have saved model to ' + path)

    # rl part ends
