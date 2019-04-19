import tensorflow as tf
import os
import sklearn.metrics
import numpy as np
import sys
import time

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.

    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class re_model:
    """Basic model class, which contains data input and tensorflow graphs, should be inherited"""
    def __init__(self, train_data_loader, batch_size, max_length=120):
        """
        class construction funciton, model initialization

        Args:
            train_data_loader: a `file_data_loader object`, which could be `npy_data_loader`
                               or `json_file_data_loader`
            batch_size: how many scopes/instances are included in one batch
            max_length: max sentence length, divide sentences into the same length (working
                        part should be finished in `data_loader`)

        Returns:
            None
        """
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.train_data_loader = train_data_loader
        self.rel_tot = train_data_loader.rel_tot
        self.word_vec_mat = train_data_loader.word_vec_mat

    def loss(self):
        """training loss, should be overrided in the subclasses"""
        raise NotImplementedError
    
    def train_logit(self):
        """training logit, should be overrided in the subclasses"""
        raise NotImplementedError
    
    def test_logit(self):
        """test logit, should be overrided in the subclasses"""
        raise NotImplementedError

class re_framework:
    """the basic training framework, does all the training and test staffs"""
    MODE_BAG = 0 # Train and test the model at bag level.
    MODE_INS = 1 # Train and test the model at instance level

    def __init__(self, train_data_loader, test_data_loader, max_length=120, batch_size=160):
        """
        class construction funciton, framework initialization

        Args:
            train_data_loader: a `file_data_loader object`, which could be `npy_data_loader`
                               or `json_file_data_loader`
            test_data_loader: similar as the `train_data_loader`
            max_length: max sentence length, divide sentences into the same length (working
                        part should be finished in `data_loader`)
            batch_size: how many scopes/instances are included in one batch

        Returns:
            None
        """
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None # default graph session

    def one_step_multi_models(self, sess, models, batch_data_gen, run_array, return_label=True):
        """
        run models and multi running tasks via session

        Args:
            sess: tf.Session() that is going to run
            models: a list. this function support multi-model training
            batch_data_gen: `data_loader` to generate batch data
            run_array: a list, contains all the running models or arrays
            return_label: boolean argument. if it is `True`, then the training label
                          will be returned either

        Returns:
            result: a tuple/list contains the result
        """
        feed_dict = {}
        batch_label = []
        for model in models:
            batch_data = batch_data_gen.next_batch(batch_data_gen.batch_size // len(models))
            feed_dict.update({
                model.word: batch_data['word'],
                model.pos1: batch_data['pos1'],
                model.pos2: batch_data['pos2'],
                model.label: batch_data['rel'],
                model.ins_label: batch_data['ins_rel'],
                model.scope: batch_data['scope'],
                model.length: batch_data['length'],
            })
            if 'mask' in batch_data and hasattr(model, "mask"): # mask data is used in PCNN models
                feed_dict.update({model.mask: batch_data['mask']})
            batch_label.append(batch_data['rel'])
        result = sess.run(run_array, feed_dict)
        batch_label = np.concatenate(batch_label)
        if return_label:
            result += [batch_label]
        return result

    def one_step(self, sess, model, batch_data, run_array):
        """
        run one model and multi running tasks via session, usually used in test operation

        Args:
            sess: tf.Session() that is going to run
            model: one model, inherited from `re_model`
            batch_data: a dict contains the batch data
            run_array: a list, contains all the running models or arrays

        Returns:
            result: a tuple/list contains the result
        """
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
            model.scope: batch_data['scope'],
            model.length: batch_data['length'],
        }
        if 'mask' in batch_data and hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask']})
        result = sess.run(run_array, feed_dict)
        return result

    def train(self, model, model_name, ckpt_dir='./checkpoint', summary_dir='./summary',
              test_result_dir='./test_result', learning_rate=0.5, max_epoch=60,
              pretrain_model=None, test_epoch=1, optimizer=tf.train.GradientDescentOptimizer,
              gpu_nums=1, not_best_stop=20):
        """
        training function

        Args:
            model: `re_model` that is going to train
            model_name: a string, to identify models, affecting checkpoint saving
            ckpt_dir: checkpoint saving directory
            summary_dir: for tensorboard use, to save summary files
            test_result_dir: directory to store the final results
            learning_rate: learning rate of optimizer
            max_epoch: how many epochs you want to train
            pretrain_model: a string, containing the checkpoint model path and model name
                            e.g. ./checkpoint/nyt_pcnn_one
            test_epoch: when do you want to test the model. default is `1`, which means 
                        test the result after every training epoch
            optimizer: training optimizer, default is `tf.train.GradientDescentOptimizer`
            gpu_nums: how many gpus you want to use when training
            not_best_stop: if there is `not_best_stop` epochs that not excel at the model
                           result, the training will be stopped

        Returns:
            None
        """
        
        assert(self.train_data_loader.batch_size % gpu_nums == 0)
        print("Start training...")
        
        # Init
        config = tf.ConfigProto(allow_soft_placement=True)  # allow cpu computing if there is no gpu available
        self.sess = tf.Session(config=config)
        optimizer = optimizer(learning_rate)
        
        # Multi GPUs
        tower_grads = []
        tower_models = []
        for gpu_id in range(gpu_nums):
            with tf.device("/gpu:%d" % gpu_id):
                with tf.name_scope("gpu_%d" % gpu_id):
                    cur_model = model(self.train_data_loader, self.train_data_loader.batch_size // gpu_nums, self.train_data_loader.max_length)
                    tower_grads.append(optimizer.compute_gradients(cur_model.loss()))
                    tower_models.append(cur_model)
                    tf.add_to_collection("loss", cur_model.loss())
                    tf.add_to_collection("train_logit", cur_model.train_logit())

        loss_collection = tf.get_collection("loss")
        loss = tf.add_n(loss_collection) / len(loss_collection)
        train_logit_collection = tf.get_collection("train_logit")
        train_logit = tf.concat(train_logit_collection, 0)

        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        """supporting check the scalars on tensorboard"""
        _output = tf.cast(tf.argmax(train_logit, -1), tf.int32) # predicted output
        _tot_acc = tf.reduce_mean(tf.cast(tf.equal(_output, tower_models[0].label), tf.float32)) # accuracy including N/A relations
        _not_na_acc = tf.reduce_mean(tf.cast(tf.logical_and(tf.equal(_output, tower_models[0].label), tf.not_equal(tower_models[0].label, 0)), tf.float32)) # accuracy not including N/A relations

        tf.summary.scalar('tot_acc', _tot_acc)
        tf.summary.scalar('not_na_acc', _not_na_acc)
        
        # Saver
        saver = tf.train.Saver(max_to_keep=None)
        if pretrain_model is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            saver.restore(self.sess, pretrain_model)

        # Training
        merged_summary = tf.summary.merge_all() # merge all scalars and histograms
        best_metric = 0
        best_prec = None
        best_recall = None
        not_best_count = 0 # Stop training after several epochs without improvement.
        global_cnt = 0  # for record summary steps
        for epoch in range(max_epoch):
            print('###### Epoch ' + str(epoch) + ' ######')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            i = 0
            time_sum = 0
            while True:
                time_start = time.time()
                try:
                    summa, iter_loss, iter_logit, _train_op, iter_label = self.one_step_multi_models(self.sess, tower_models, self.train_data_loader, [merged_summary, loss, train_logit, train_op])
                except StopIteration:
                    break
                summary_writer.add_summary(summa, global_cnt)
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)
                iter_correct = (iter_output == iter_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += iter_label.shape[0]
                tot_not_na += (iter_label != 0).sum()
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

            if not_best_count >= not_best_stop:
                break

            global_cnt += 1

        print("######")
        print("Finish training " + model_name)
        print("Best epoch auc = %f" % (best_metric))
        if (not best_prec is None) and (not best_recall is None):
            if not os.path.isdir(test_result_dir):
                os.mkdir(test_result_dir)
            np.save(os.path.join(test_result_dir, model_name + "_x.npy"), best_recall)
            np.save(os.path.join(test_result_dir, model_name + "_y.npy"), best_prec)

    def test(self, model, ckpt=None, return_result=False, mode=MODE_BAG):
        """
        test function, to evaluate model

        Args:
            model: a `re_model` which has not been instantiated
            ckpt: whether there is a pretained checkpoing model
            return_result: if True, the predicted result will be returned, either
            mode: basically it is at the bag level
        
        Returns:
            auc: if return_result is True, return AUC and predicted labels, 
                 else return AUC only
        """
        if mode == re_framework.MODE_BAG:
            return self.__test_bag__(model, ckpt=ckpt, return_result=return_result)
        elif mode == re_framework.MODE_INS:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    def __test_bag__(self, model, ckpt=None, return_result=False):
        """
        test function at bag level

        Args:
            model: a `re_model` which has not been instantiated
            ckpt: whether there is a pretained checkpoing model
            return_result: if True, the predicted result will be returned, either
        
        Returns:
            auc: if return_result is True, return AUC and predicted labels, 
                 else return AUC only
        """
        print("Testing...")
        if self.sess == None:
            self.sess = tf.Session()
        model = model(self.test_data_loader, self.test_data_loader.batch_size, self.test_data_loader.max_length)
        if not ckpt is None:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt)
        tot_correct = 0
        tot_not_na_correct = 0
        tot = 0
        tot_not_na = 0
        entpair_tot = 0
        test_result = []
        pred_result = []
         
        for i, batch_data in enumerate(self.test_data_loader):
            iter_logit = self.one_step(self.sess, model, batch_data, [model.test_logit()])[0]
            iter_output = iter_logit.argmax(-1)
            iter_correct = (iter_output == batch_data['rel']).sum()
            iter_not_na_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] != 0).sum()
            tot_correct += iter_correct
            tot_not_na_correct += iter_not_na_correct
            tot += batch_data['rel'].shape[0]
            tot_not_na += (batch_data['rel'] != 0).sum()
            if tot_not_na > 0:
                sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (i, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()
            for idx in range(len(iter_logit)):
                for rel in range(1, self.test_data_loader.rel_tot):
                    test_result.append({'score': iter_logit[idx][rel], 'flag': batch_data['multi_rel'][idx][rel]})
                    if batch_data['entpair'][idx] != "None#None":
                        pred_result.append({'score': float(iter_logit[idx][rel]), 'entpair': batch_data['entpair'][idx].encode('utf-8'), 'relation': rel})
                entpair_tot += 1 
        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        prec = []
        recall = [] 
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / self.test_data_loader.relfact_tot)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        print("\n[TEST] auc: {}".format(auc))
        print("Finish testing")
        self.cur_prec = prec
        self.cur_recall = recall

        if not return_result:
            return auc
        else:
            return (auc, pred_result)
