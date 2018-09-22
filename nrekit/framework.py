import tensorflow as tf
import os
import sklearn.metrics
import numpy as np
import sys

class re_framework:
    def __init__(self, train_data_loader, test_data_loader, max_length=120, batch_size=160):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        self.word_vec_mat = train_data_loader.word_vec_mat
        self.rel_tot = train_data_loader.rel_tot
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None

    def one_step(self, sess, batch_data, run_array, keep_prob=1.0):
        feed_dict = {
            self.word: batch_data['word'],
            self.pos1: batch_data['pos1'],
            self.pos2: batch_data['pos2'],
            self.label: batch_data['rel'],
            self.scope: batch_data['scope'],
            self.length: batch_data['length'],
            self.keep_prob: keep_prob
        }
        result = sess.run(run_array, feed_dict)
        return result
   
    def train(self,
              loss,
              train_output,
              test_output,
              ckpt_dir,
              model_name='model',
              summary_dir='./summary',
              learning_rate=0.1,
              max_epoch=60,
              pretrain_model=None,
              test_epoch=1,
              optimizer=tf.train.GradientDescentOptimizer):

        print("Start training...")
        
        # Init
        self.sess = tf.Session()
        optimizer = optimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

        # Summary
        tf.summary.scalar('learning_rate', learning_rate)
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        # Saver
        saver = tf.train.Saver(max_to_keep=None)
        if pretrain_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            saver.restore(self.sess, pretrain_model)

        # Training
        best_metric = 0
        for epoch in range(max_epoch):
            print('Epoch ' + str(epoch) + ' starts...')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            for i, batch_data in enumerate(self.train_data_loader):
                iter_loss, iter_output, _train_op  = self.one_step(self.sess, batch_data, [loss, train_output, train_op], keep_prob=0.5)
                iter_correct = (iter_output == batch_data['rel']).sum()
                iter_not_na_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += batch_data['rel'].shape[0]
                tot_not_na += (batch_data['rel'] != 0).shape[0]
                sys.stdout.write("epoch %d step %d | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (epoch, i, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()

            if (epoch + 1) % test_epoch == 0:
                metric = self.test(test_output)
                if metric > best_metric:
                    print("Best model, storing...")
                    if not os.path.isdir(ckpt_dir):
                        os.mkdir(ckpt_dir)
                    path = saver.save(self.sess, os.path.join(ckpt_dir, model_name))
                    print("Finish storing")
    
    def test(self,
             output,
             ckpt=None,
             eval_by_accuracy=False):
        print("\nTesting...")
        if self.sess == None:
            self.sess = tf.Session()
        if not ckpt is None:
            raise NotImplementedError
        tot_correct = 0
        tot = 0
        entpair_tot = 0
        test_result = []
        for i, batch_data in enumerate(self.test_data_loader):
            iter_output = self.one_step(self.sess, batch_data, [output], keep_prob=1.0)
            iter_correct = (iter_output == batch_data['rel']).sum()
            tot_correct += iter_correct
            tot += batch_data['rel'].shape[0]
            sys.stdout.write("[TEST] step %d | accuracy: %f\r" % (i, float(tot_correct) / tot))
            sys.stdout.flush()
            for idx in range(len(iter_output)):
                for rel in range(1, self.test_data_loader.rel_tot):
                    test_result.append({'score': iter_output[idx][rel], 'flag': batch_data['multi_rel'][idx][rel]})
                entpair_tot += 1
            
        if eval_by_accuracy:
            return float(tot_correct) / tot

        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        prec = []
        recall = [] 
        correct = 0
        for i, item in enumerate(sorted_test_result):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / self.test_data_loader.relfact_tot)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        print("\n[TEST] auc: {}".format(auc))
        print("Finish testing")
        return auc 
