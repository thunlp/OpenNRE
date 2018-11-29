import tensorflow as tf
import os
import sklearn.metrics
import numpy as np
import sys
import math
import time
import framework
import network

class policy_agent(framework.re_model):
    def __init__(self, train_data_loader, batch_size, max_length=120):
        framework.re_model.__init__(self, train_data_loader, batch_size, max_length)
        self.weights = tf.placeholder(tf.float32, shape=(), name="weights_scalar")

        x = network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)
        x_train = network.encoder.cnn(x, keep_prob=0.5)
        x_test = network.encoder.cnn(x, keep_prob=1.0)
        self._train_logit = network.selector.instance(x_train, 2, keep_prob=0.5)
        self._test_logit = network.selector.instance(x_test, 2, keep_prob=1.0)
        self._loss = network.classifier.softmax_cross_entropy(self._train_logit, self.ins_label, 2, weights=self.weights)

    def loss(self):
        return self._loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit

class rl_re_framework(framework.re_framework):
    def __init__(self, train_data_loader, test_data_loader, max_length=120, batch_size=160):
        framework.re_framework.__init__(self, train_data_loader, test_data_loader, max_length, batch_size)

    def agent_one_step(self, sess, agent_model, batch_data, run_array, weights=1):
        feed_dict = {
            agent_model.word: batch_data['word'],
            agent_model.pos1: batch_data['pos1'],
            agent_model.pos2: batch_data['pos2'],
            agent_model.ins_label: batch_data['agent_label'],
            agent_model.length: batch_data['length'],
            agent_model.weights: weights
        }
        if 'mask' in batch_data and hasattr(agent_model, "mask"):
            feed_dict.update({agent_model.mask: batch_data['mask']})
        result = sess.run(run_array, feed_dict)
        return result

    def pretrain_main_model(self, max_epoch):
        for epoch in range(max_epoch):
            print('###### Epoch ' + str(epoch) + ' ######')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            i = 0
            time_sum = 0
            
            for i, batch_data in enumerate(self.train_data_loader):
                time_start = time.time()
                iter_loss, iter_logit, _train_op = self.one_step(self.sess, self.model, batch_data, [self.model.loss(), self.model.train_logit(), self.train_op])
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)
                iter_label = batch_data['rel']
                iter_correct = (iter_output == iter_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += iter_label.shape[0]
                tot_not_na += (iter_label != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("[pretrain main model] epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()
                i += 1
            print("\nAverage iteration time: %f" % (time_sum / i))

    def pretrain_agent_model(self, max_epoch):
        # Pre-train policy agent
        for epoch in range(max_epoch):
            print('###### [Pre-train Policy Agent] Epoch ' + str(epoch) + ' ######')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            time_sum = 0
            
            for i, batch_data in enumerate(self.train_data_loader):
                time_start = time.time()
                batch_data['agent_label'] = batch_data['ins_rel'] + 0
                batch_data['agent_label'][batch_data['agent_label'] > 0] = 1
                iter_loss, iter_logit, _train_op = self.agent_one_step(self.sess, self.agent_model, batch_data, [self.agent_model.loss(), self.agent_model.train_logit(), self.agent_train_op])
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)
                iter_label = batch_data['ins_rel']
                iter_correct = (iter_output == iter_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += iter_label.shape[0]
                tot_not_na += (iter_label != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("[pretrain policy agent] epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()
                i += 1

    def train(self,
              model, # The main model
              agent_model, # The model of policy agent
              model_name,
              ckpt_dir='./checkpoint',
              summary_dir='./summary',
              test_result_dir='./test_result',
              learning_rate=0.5,
              max_epoch=60,
              pretrain_agent_epoch=1,
              pretrain_model=None,
              test_epoch=1,
              optimizer=tf.train.GradientDescentOptimizer):
        
        print("Start training...")
        
        # Init
        self.model = model(self.train_data_loader, self.train_data_loader.batch_size, self.train_data_loader.max_length)
        model_optimizer = optimizer(learning_rate)
        grads = model_optimizer.compute_gradients(self.model.loss())
        self.train_op = model_optimizer.apply_gradients(grads)

        # Init policy agent
        self.agent_model = agent_model(self.train_data_loader, self.train_data_loader.batch_size, self.train_data_loader.max_length)
        agent_optimizer = optimizer(learning_rate)
        agent_grads = agent_optimizer.compute_gradients(self.agent_model.loss())
        self.agent_train_op = agent_optimizer.apply_gradients(agent_grads)

        # Session, writer and saver
        self.sess = tf.Session()
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        saver = tf.train.Saver(max_to_keep=None)
        if pretrain_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            saver.restore(self.sess, pretrain_model)

        self.pretrain_main_model(max_epoch=5) # Pre-train main model
        self.pretrain_agent_model(max_epoch=1) # Pre-train policy agent 

        # Train
        tot_delete = 0
        batch_count = 0
        instance_count = 0
        reward = 0.0
        best_metric = 0
        best_prec = None
        best_recall = None
        not_best_count = 0 # Stop training after several epochs without improvement.
        for epoch in range(max_epoch):
            print('###### Epoch ' + str(epoch) + ' ######')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            i = 0
            time_sum = 0
            batch_stack = []
           
            # Update policy agent
            for i, batch_data in enumerate(self.train_data_loader):
                # Make action
                batch_data['agent_label'] = batch_data['ins_rel'] + 0
                batch_data['agent_label'][batch_data['agent_label'] > 0] = 1
                batch_stack.append(batch_data)
                iter_logit = self.agent_one_step(self.sess, self.agent_model, batch_data, [self.agent_model.train_logit()])[0]
                action_result = iter_logit.argmax(-1)
                
                # Calculate reward
                batch_delete = np.sum(np.logical_and(batch_data['ins_rel'] != 0, action_result == 0))
                batch_data['ins_rel'][action_result == 0] = 0
                iter_loss = self.one_step(self.sess, self.model, batch_data, [self.model.loss()])[0]
                reward += iter_loss
                tot_delete += batch_delete
                batch_count += 1

                # Update parameters of policy agent
                alpha = 0.1
                if batch_count == 100:
                    reward = reward / float(batch_count)
                    average_loss = reward
                    reward = - math.log(1 - math.e ** (-reward))
                    sys.stdout.write('tot delete : %f | reward : %f | average loss : %f\r' % (tot_delete, reward, average_loss))
                    sys.stdout.flush()
                    for batch_data in batch_stack:
                        self.agent_one_step(self.sess, self.agent_model, batch_data, [self.agent_train_op], weights=reward * alpha)
                    batch_count = 0
                    reward = 0
                    tot_delete = 0
                    batch_stack = []
                i += 1

            # Train the main model
            for i, batch_data in enumerate(self.train_data_loader):
                batch_data['agent_label'] = batch_data['ins_rel'] + 0
                batch_data['agent_label'][batch_data['agent_label'] > 0] = 1
                time_start = time.time()

                # Make actions
                iter_logit = self.agent_one_step(self.sess, self.agent_model, batch_data, [self.agent_model.train_logit()])[0]
                action_result = iter_logit.argmax(-1)
                batch_data['ins_rel'][action_result == 0] = 0
                
                # Real training
                iter_loss, iter_logit, _train_op = self.agent_one_step(self.sess, self.agent_model, batch_data, [self.agent_model.loss(), self.agent_model.train_logit(), self.agent_train_op])
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)
                if tot_not_na > 0:
                    sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()
                i += 1
            print("\nAverage iteration time: %f" % (time_sum / i))

            if (epoch + 1) % test_epoch == 0:
                metric = self.test(model)
                if metric > best_metric:
                    best_metric = metric
                    best_prec = self.cur_prec
                    best_recall = self.cur_recall
                    print("Best model, storing...")
                    if not os.path.isdir(ckpt_dir):
                        os.mkdir(ckpt_dir)
                    path = saver.save(self.sess, os.path.join(ckpt_dir, model_name))
                    print("Finish storing")
                    not_best_count = 0
                else:
                    not_best_count += 1

            if not_best_count >= 20:
                break

        print("######")
        print("Finish training " + model_name)
        print("Best epoch auc = %f" % (best_metric))
        if (not best_prec is None) and (not best_recall is None):
            if not os.path.isdir(test_result_dir):
                os.mkdir(test_result_dir)
            np.save(os.path.join(test_result_dir, model_name + "_x.npy"), best_recall)
            np.save(os.path.join(test_result_dir, model_name + "_y.npy"), best_prec)

