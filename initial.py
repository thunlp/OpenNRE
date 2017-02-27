import numpy as np
import os

#embedding the position 
def pos_embed(x):
	if x < -60:
		return 0
	if x >= -60 and x <= 60:
		return x+61
	if x > 60:
		return 122

#reading data
def init():
	
	print 'reading word embedding data...'
	vec = []
	word2id = {}
	f = open('./origin_data/vec.txt')
	f.readline()
	while True:
		content = f.readline()
		if content == '':
			break
		content = content.strip().split()
		word2id[content[0]] = len(word2id)
		content = content[1:]
		content = [(float)(i) for i in content]
		vec.append(content)
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	
	dim = 50
	vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
	vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
	vec = np.array(vec,dtype=np.float32)

	
	print 'reading relation to id'
	relation2id = {}	
	f = open('./origin_data/relation2id.txt','r')
	while True:
		content = f.readline()
		if content == '':
			break
		content = content.strip().split()
		relation2id[content[0]] = int(content[1])
	f.close()

	#length of sentence is 70
	fixlen = 70
	#max length of position embedding is 60 (-60~+60)
	maxlen = 60

	train_sen = {} #{entity pair:[[sentence 1],[sentence 2]...]}
	train_ans = {} #{entity pair:label} the label is one-hot vector


	print 'reading train data...'
	f = open('./origin_data/train.txt','r')

	while True:
		content = f.readline()
		if content == '':
			break
		
		content = content.strip().split()
		#get entity name
		en1 = content[2] 
		en2 = content[3]

		#put the same entity pair sentences into a dict
		tup = (en1,en2)
		if tup not in train_sen:
			train_sen[tup]=[]
			y_id = relation2id[content[4]]
			label = [0 for i in range(len(relation2id))]
			label[y_id] = 1
			train_ans[tup] = label
		sentence = content[5:-1]
		
		en1pos = 0
		en2pos = 0
		
		for i in range(len(sentence)):
			if sentence[i] == en1:
				en1pos = i
			if sentence[i] == en2:
				en2pos = i
		output = []

		for i in range(fixlen):
			word = word2id['BLANK']
			rel_e1 = pos_embed(i - en1pos)
			rel_e2 = pos_embed(i - en2pos)
			output.append([word,rel_e1,rel_e2])

		for i in range(min(fixlen,len(sentence))):
			word = 0
			if sentence[i] not in word2id:
				word = word2id['UNK']
			else:
				word = word2id[sentence[i]]
			
			output[i][0] = word

		train_sen[tup].append(output)

	print('reading test data ...')

	test_sen = {}
	test_ans = {}

	f = open('./origin_data/test.txt','r')

	while True:
		content = f.readline()
		if content == '':
			break
		
		content = content.strip().split()
		en1 = content[2]
		en2 = content[3]		
		tup = (en1,en2)
		if tup not in test_sen:
			test_sen[tup]=[]
			y_id = relation2id[content[4]]
			label = [0 for i in range(len(relation2id))]
			label[y_id] = 1
			test_ans[tup] = label
			
		sentence = content[5:-1]

		en1pos = 0
		en2pos = 0
		
		for i in range(len(sentence)):
			if sentence[i] == en1:
				en1pos = i
			if sentence[i] == en2:
				en2pos = i
		output = []

		for i in range(fixlen):
			word = word2id['BLANK']
			rel_e1 = pos_embed(i - en1pos)
			rel_e2 = pos_embed(i - en2pos)
			output.append([word,rel_e1,rel_e2])

		for i in range(min(fixlen,len(sentence))):
			word = 0
			if sentence[i] not in word2id:
				word = word2id['UNK']
			else:
				word = word2id[sentence[i]]

			output[i][0] = word
		test_sen[tup].append(output)
	
	train_x = []
	train_y = []
	test_x = []
	test_y = []

	print 'organizing train data'
	f = open('./data/train_q&a.txt','w')
	temp = 0
	for i in train_sen:
		train_x.append(train_sen[i])
		train_y.append(train_ans[i])
		f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(train_ans[i]))+'\n')
		temp+=1
	f.close()

	print 'organizing test data'
	id2tup = []
	temp = 0
	f = open('./data/test_q&a.txt','w')
	for i in test_sen:
		test_x.append(test_sen[i])
		test_y.append(test_ans[i])
		id2tup.append(i)
		f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(test_ans[i]))+'\n')
		temp+=1
	f.close()

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	

	np.save('./data/vec.npy',vec)
	np.save('./data/train_x.npy',train_x)
	np.save('./data/train_y.npy',train_y)
	np.save('./data/testall_x.npy',test_x)
	np.save('./data/testall_y.npy',test_y)

	#get test data for P@N evaluation, in which only entity pairs with more than 1 sentence exist
	print 'get test data for p@n test'
	
	pone_test_x = []
	pone_test_y = []

	ptwo_test_x = []
	ptwo_test_y = []
	
	pall_test_x = []
	pall_test_y = []

	for i in range(len(test_x)):
		if len(test_x[i]) > 1:
			
			pall_test_x.append(test_x[i])
			pall_test_y.append(test_y[i])
			
			onetest = []
			temp = np.random.randint(len(test_x[i]))
			onetest.append(test_x[i][temp])
			pone_test_x.append(onetest)
			pone_test_y.append(test_y[i])

			twotest = []
			temp1 = np.random.randint(len(test_x[i]))
			temp2 = np.random.randint(len(test_x[i]))
			while temp1 == temp2:
				temp2 = np.random.randint(len(test_x[i]))
			twotest.append(test_x[i][temp1])
			twotest.append(test_x[i][temp2])
			ptwo_test_x.append(twotest)
			ptwo_test_y.append(test_y[i])

	pone_test_x = np.array(pone_test_x)
	pone_test_y = np.array(pone_test_y)
	ptwo_test_x = np.array(ptwo_test_x)
	ptwo_test_y = np.array(ptwo_test_y)	
	pall_test_x = np.array(pall_test_x)
	pall_test_y = np.array(pall_test_y)

	np.save('./data/pone_test_x.npy',pone_test_x)
	np.save('./data/pone_test_y.npy',pone_test_y)
	np.save('./data/ptwo_test_x.npy',ptwo_test_x)
	np.save('./data/ptwo_test_y.npy',ptwo_test_y)
	np.save('./data/pall_test_x.npy',pall_test_x)
	np.save('./data/pall_test_y.npy',pall_test_y)

def seperate():
	
	print 'reading training data'
	x_train = np.load('./data/train_x.npy')

	train_word = []
	train_pos1 = []
	train_pos2 = []

	print 'seprating train data'
	for i in range(len(x_train)):
		word = []
		pos1 = []
		pos2 = []
		for j in x_train[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
		train_word.append(word)
		train_pos1.append(pos1)
		train_pos2.append(pos2)

	train_word = np.array(train_word)
	train_pos1 = np.array(train_pos1)
	train_pos2 = np.array(train_pos2)
	np.save('./data/train_word.npy',train_word)
	np.save('./data/train_pos1.npy',train_pos1)
	np.save('./data/train_pos2.npy',train_pos2)

	print 'reading p-one test data'
	x_test = np.load('./data/pone_test_x.npy')
	print 'seperating p-one test data'
	test_word = []
	test_pos1 = []
	test_pos2 = []

	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []
		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)

	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)
	np.save('./data/pone_test_word.npy',test_word)
	np.save('./data/pone_test_pos1.npy',test_pos1)
	np.save('./data/pone_test_pos2.npy',test_pos2)

	print 'reading p-two test data'
	x_test = np.load('./data/ptwo_test_x.npy')
	print 'seperating p-two test data'
	test_word = []
	test_pos1 = []
	test_pos2 = []

	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []
		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)

	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)
	np.save('./data/ptwo_test_word.npy',test_word)
	np.save('./data/ptwo_test_pos1.npy',test_pos1)
	np.save('./data/ptwo_test_pos2.npy',test_pos2)

	print 'reading p-all test data'
	x_test = np.load('./data/pall_test_x.npy')
	print 'seperating p-all test data'
	test_word = []
	test_pos1 = []
	test_pos2 = []

	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []
		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)

	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)
	np.save('./data/pall_test_word.npy',test_word)
	np.save('./data/pall_test_pos1.npy',test_pos1)
	np.save('./data/pall_test_pos2.npy',test_pos2)


	print 'seperating test all data'
	x_test = np.load('./data/testall_x.npy')

	test_word = []
	test_pos1 = []
	test_pos2 = []

	for i in range(len(x_test)):
		word = []
		pos1 = []
		pos2 = []
		for j in x_test[i]:
			temp_word = []
			temp_pos1 = []
			temp_pos2 = []
			for k in j:
				temp_word.append(k[0])
				temp_pos1.append(k[1])
				temp_pos2.append(k[2])
			word.append(temp_word)
			pos1.append(temp_pos1)
			pos2.append(temp_pos2)
		test_word.append(word)
		test_pos1.append(pos1)
		test_pos2.append(pos2)



	test_word = np.array(test_word)
	test_pos1 = np.array(test_pos1)
	test_pos2 = np.array(test_pos2)

	np.save('./data/testall_word.npy',test_word)
	np.save('./data/testall_pos1.npy',test_pos1)
	np.save('./data/testall_pos2.npy',test_pos2)


def getsmall():
	
 	print 'reading training data'
	word = np.load('./data/train_word.npy')
	pos1 = np.load('./data/train_pos1.npy')
	pos2 = np.load('./data/train_pos2.npy')
	y = np.load('./data/train_y.npy')

	new_word = []
	new_pos1 = []
	new_pos2 = []
	new_y = []

	#we slice some big batch in train data into small batches in case of running out of memory
	print 'get small training data'
	for i in range(len(word)):
		lenth = len(word[i])
		if lenth <= 1000:
			new_word.append(word[i])
			new_pos1.append(pos1[i])
			new_pos2.append(pos2[i])
			new_y.append(y[i])

		if lenth > 1000 and lenth < 2000:
			
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:])
			
			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:])

			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:])
			
			new_y.append(y[i])
			new_y.append(y[i])

		
		if lenth > 2000 and lenth < 3000:
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:2000])
			new_word.append(word[i][2000:])
			
			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:2000])
			new_pos1.append(pos1[i][2000:])
			
			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:2000])
			new_pos2.append(pos2[i][2000:])
			
			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])

		if lenth > 3000 and lenth < 4000:
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:2000])
			new_word.append(word[i][2000:3000])
			new_word.append(word[i][3000:])
		
			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:2000])
			new_pos1.append(pos1[i][2000:3000])
			new_pos1.append(pos1[i][3000:])

			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:2000])
			new_pos2.append(pos2[i][2000:3000])
			new_pos2.append(pos2[i][3000:])

			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])

		if lenth > 4000:
			
			new_word.append(word[i][:1000])
			new_word.append(word[i][1000:2000])
			new_word.append(word[i][2000:3000])
			new_word.append(word[i][3000:4000])
			new_word.append(word[i][4000:])

			new_pos1.append(pos1[i][:1000])
			new_pos1.append(pos1[i][1000:2000])
			new_pos1.append(pos1[i][2000:3000])
			new_pos1.append(pos1[i][3000:4000])
			new_pos1.append(pos1[i][4000:])

			new_pos2.append(pos2[i][:1000])
			new_pos2.append(pos2[i][1000:2000])
			new_pos2.append(pos2[i][2000:3000])
			new_pos2.append(pos2[i][3000:4000])
			new_pos2.append(pos2[i][4000:])

			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])
			new_y.append(y[i])


	new_word = np.array(new_word)
	new_pos1 = np.array(new_pos1)
	new_pos2 = np.array(new_pos2)
	new_y = np.array(new_y)

	np.save('./data/small_word.npy',new_word)
	np.save('./data/small_pos1.npy',new_pos1)
	np.save('./data/small_pos2.npy',new_pos2)
	np.save('./data/small_y.npy',new_y)

#get answer metric for PR curve evaluation
def getans():
	test_y = np.load('./data/testall_y.npy')
	eval_y = []
	for i in test_y:
		eval_y.append(i[1:])
	allans = np.reshape(eval_y,(-1))
	np.save('./data/allans.npy',allans)

init()
seperate()
getsmall()
getans()


