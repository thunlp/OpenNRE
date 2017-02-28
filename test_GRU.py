import tensorflow as tf
import numpy as np
import time
import datetime
import os
from sklearn.metrics import average_precision_score

class Settings(object):

	def __init__(self):

		self.vocab_size = 114042
		self.num_steps = 70
		self.num_epochs = 100
		self.num_classes = 58
		self.gru_size = 230
		self.keep_prob = 0.5
		self.num_layers = 1
		self.pos_size = 5
		self.pos_num = 123
		self.big_num = 50

class GRU:
	def __init__(self,is_training,word_embeddings,settings):
		
		self.num_steps = num_steps = settings.num_steps
	
		self.vocab_size = vocab_size = settings.vocab_size
		self.num_classes = num_classes = settings.num_classes
		self.gru_size = gru_size = settings.gru_size
		self.big_num = big_num = settings.big_num

		self.input_word = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='input_word')
		self.input_pos1 = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='input_pos1')
		self.input_pos2 = tf.placeholder(dtype=tf.int32,shape=[None,num_steps],name='input_pos2')
		self.input_y = tf.placeholder(dtype=tf.float32,shape=[None,num_classes],name='input_y')
		self.total_shape = tf.placeholder(dtype=tf.int32,shape=[big_num+1],name='total_shape')
		total_num = self.total_shape[-1]

		#word_embedding = tf.get_variable(initializer=word_embeddings,name = 'word_embedding')
		word_embedding = tf.get_variable('word_embedding',[settings.vocab_size,50])		
		pos1_embedding = tf.get_variable('pos1_embedding',[settings.pos_num,settings.pos_size])
		pos2_embedding = tf.get_variable('pos2_embedding',[settings.pos_num,settings.pos_size])

		attention_w = tf.get_variable('attention_omega',[gru_size,1])
		sen_a = tf.get_variable('attention_A',[gru_size])
		sen_r = tf.get_variable('query_r',[gru_size,1])		
		relation_embedding = tf.get_variable('relation_embedding',[self.num_classes,gru_size])
		sen_d = tf.get_variable('bias_d',[self.num_classes])

		gru_cell_forward = tf.nn.rnn_cell.GRUCell(gru_size)
		gru_cell_backward = tf.nn.rnn_cell.GRUCell(gru_size)


		if is_training and settings.keep_prob < 1:
			gru_cell_forward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_forward,output_keep_prob=settings.keep_prob)
			gru_cell_backward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_backward,output_keep_prob=settings.keep_prob)
		
		cell_forward = tf.nn.rnn_cell.MultiRNNCell([gru_cell_forward]*settings.num_layers)
		cell_backward = tf.nn.rnn_cell.MultiRNNCell([gru_cell_backward]*settings.num_layers)		

		sen_repre = []
		sen_alpha = []
		sen_s = []
		sen_out = []
		self.prob = []
		self.predictions = []
		self.loss = []
		self.accuracy = []
		self.total_loss = 0.0

		self._initial_state_forward = cell_forward.zero_state(total_num,tf.float32)
		self._initial_state_backward = cell_backward.zero_state(total_num,tf.float32)

		inputs_forward = tf.concat(2,[tf.nn.embedding_lookup(word_embedding,self.input_word),tf.nn.embedding_lookup(pos1_embedding,self.input_pos1),tf.nn.embedding_lookup(pos2_embedding,self.input_pos2)])
		inputs_backward = tf.concat(2,[tf.nn.embedding_lookup(word_embedding,tf.reverse(self.input_word,[False,True])),tf.nn.embedding_lookup(pos1_embedding,tf.reverse(self.input_pos1,[False,True])),tf.nn.embedding_lookup(pos1_embedding,tf.reverse(self.input_pos2,[False,True]))])
		
		outputs_forward = []

		state_forward = self._initial_state_forward

		with tf.variable_scope('GRU_FORWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_forward,state_forward) = cell_forward(inputs_forward[:,step,:],state_forward)
				outputs_forward.append(cell_output_forward)				
		
		outputs_backward = []

		state_backward = self._initial_state_backward
		with tf.variable_scope('GRU_BACKWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_backward,state_backward) = cell_backward(inputs_backward[:,step,:],state_backward)
				outputs_backward.append(cell_output_backward)
		
		output_forward = tf.reshape(tf.concat(1,  outputs_forward), [total_num, num_steps, gru_size])
		output_backward  = tf.reverse(tf.reshape(tf.concat(1,  outputs_backward), [total_num, num_steps, gru_size]), [False, True, False])
		
		output_h = tf.add(output_forward,output_backward)
		
		attention_r = tf.reshape(tf.batch_matmul(tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h),[total_num*num_steps,gru_size]),attention_w),[total_num,num_steps])),[total_num,1,num_steps]),output_h),[total_num,gru_size])

		
		for i in range(big_num):

			sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i+1]]))
			batch_size = self.total_shape[i+1]-self.total_shape[i]
		
			sen_alpha.append(tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.mul(sen_repre[i],sen_a),sen_r),[batch_size])),[1,batch_size]))
		
			sen_s.append(tf.reshape(tf.matmul(sen_alpha[i],sen_repre[i]),[gru_size,1]))
			sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding,sen_s[i]),[self.num_classes]),sen_d))
		
			self.prob.append(tf.nn.softmax(sen_out[i]))

			with tf.name_scope("output"):
				self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

			with tf.name_scope("loss"):
				self.loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(sen_out[i], self.input_y[i])))
				if i == 0:
					self.total_loss = self.loss[i]
				else:
					self.total_loss += self.loss[i]


			with tf.name_scope("accuracy"):
				self.accuracy.append(tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"), name="accuracy"))


def main(_):

	# ATTENTION: change pathname before you load your model
	pathname = "./model/ATT_GRU_model-"
	

	test_settings = Settings()
	test_settings.vocab_size = 114044
	test_settings.num_classes = 58	
	test_settings.big_num = 262

	big_num_test = test_settings.big_num

	with tf.Graph().as_default():

		sess = tf.Session()
		with sess.as_default():

			def test_step(word_batch, pos1_batch, pos2_batch, y_batch):
	
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

				feed_dict[mtest.total_shape] = total_shape
				feed_dict[mtest.input_word] = total_word
				feed_dict[mtest.input_pos1] = total_pos1
				feed_dict[mtest.input_pos2] = total_pos2
				feed_dict[mtest.input_y] = y_batch

				loss, accuracy ,prob= sess.run(
					[mtest.loss, mtest.accuracy,mtest.prob], feed_dict)
				return prob,accuracy

			# evaluate p@n
			def eval_pn(test_y,test_word,test_pos1,test_pos2,test_settings):
				allprob = [] 
				acc = []
				for i in range(int(len(test_word)/float(test_settings.big_num))):
					prob,accuracy = test_step(test_word[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos1[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos2[i*test_settings.big_num:(i+1)*test_settings.big_num],test_y[i*test_settings.big_num:(i+1)*test_settings.big_num])
					acc.append(np.mean(np.reshape(np.array(accuracy),(test_settings.big_num))))
					prob = np.reshape(np.array(prob),(test_settings.big_num,test_settings.num_classes))
					for single_prob in prob:
						allprob.append(single_prob[1:])
				allprob = np.reshape(np.array(allprob),(-1))
				eval_y = []
				for i in test_y:
					eval_y.append(i[1:])
				allans = np.reshape(eval_y,(-1))
				order = np.argsort(-allprob)

				print 'P@100:'
				top100 = order[:100]
				correct_num_100 = 0.0
				for i in top100:
					if allans[i] == 1:
						correct_num_100 += 1.0
				print correct_num_100/100

				print 'P@200:'
				top200 = order[:200]
				correct_num_200 = 0.0
				for i in top200:
					if allans[i] == 1:
						correct_num_200 += 1.0
				print correct_num_200/200

				print 'P@300:'
				top300 = order[:300]
				correct_num_300 = 0.0
				for i in top300:
					if allans[i] == 1:
						correct_num_300 += 1.0
				print correct_num_300/300

			
			with tf.variable_scope("model"):
				mtest = GRU(is_training=False, word_embeddings = None, settings = test_settings)

			saver = tf.train.Saver()
			

			# ATTENTION: change the list to the iters you want to test !!

			for model_iter in [11400]:

				saver.restore(sess,pathname+str(model_iter))
				print("Evaluating P@N for iter "+str(model_iter))

				print 'Evaluating P@N for one'
				test_y = np.load('./data/pone_test_y.npy')
				test_word = np.load('./data/pone_test_word.npy')
				test_pos1 = np.load('./data/pone_test_pos1.npy')
				test_pos2 = np.load('./data/pone_test_pos2.npy')
				eval_pn(test_y,test_word,test_pos1,test_pos2,test_settings)

				print 'Evaluating P@N for two'
				test_y = np.load('./data/ptwo_test_y.npy')
				test_word = np.load('./data/ptwo_test_word.npy')
				test_pos1 = np.load('./data/ptwo_test_pos1.npy')
				test_pos2 = np.load('./data/ptwo_test_pos2.npy')
				eval_pn(test_y,test_word,test_pos1,test_pos2,test_settings)

				print 'Evaluating P@N for all'
				test_y = np.load('./data/pall_test_y.npy')
				test_word = np.load('./data/pall_test_word.npy')
				test_pos1 = np.load('./data/pall_test_pos1.npy')
				test_pos2 = np.load('./data/pall_test_pos2.npy')
				eval_pn(test_y,test_word,test_pos1,test_pos2,test_settings)
				
				time_str = datetime.datetime.now().isoformat()
				print time_str
				print 'Evaluating all test data and save data for PR curve'
				test_y = np.load('./data/testall_y.npy')
				test_word = np.load('./data/testall_word.npy')
				test_pos1 = np.load('./data/testall_pos1.npy')
				test_pos2 = np.load('./data/testall_pos2.npy')
				allprob = [] 
				acc = []
				for i in range(int(len(test_word)/float(test_settings.big_num))):
					prob,accuracy = test_step(test_word[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos1[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos2[i*test_settings.big_num:(i+1)*test_settings.big_num],test_y[i*test_settings.big_num:(i+1)*test_settings.big_num])
					acc.append(np.mean(np.reshape(np.array(accuracy),(test_settings.big_num))))
					prob = np.reshape(np.array(prob),(test_settings.big_num,test_settings.num_classes))
					for single_prob in prob:
						allprob.append(single_prob[1:])
				allprob = np.reshape(np.array(allprob),(-1))
				order = np.argsort(-allprob)

				print 'saving all test result...'
				current_step = model_iter
				
				# ATTENTION: change the save path before you save your result !!
				np.save('./out/sample_allprob_iter_'+str(current_step)+'.npy',allprob)
				allans = np.load('./data/allans.npy')
				
				#caculate the pr curve area
				average_precision = average_precision_score(allans,allprob)
				print 'PR curve area:'+str(average_precision)
				time_str = datetime.datetime.now().isoformat()
				print time_str
				print 'P@N for all test data:'
				print 'P@100:'
				top100 = order[:100]
				correct_num_100 = 0.0
				for i in top100:
					if allans[i] == 1:
						correct_num_100 += 1.0
				print correct_num_100/100

				print 'P@200:'
				top200 = order[:200]
				correct_num_200 = 0.0
				for i in top200:
					if allans[i] == 1:
						correct_num_200 += 1.0
				print correct_num_200/200

				print 'P@300:'
				top300 = order[:300]
				correct_num_300 = 0.0
				for i in top300:
					if allans[i] == 1:
						correct_num_300 += 1.0
				print correct_num_300/300

if __name__ == "__main__":
	tf.app.run() 
