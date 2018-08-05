from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_att_adv(is_training):
    if is_training:
        with tf.variable_scope('pcnn_att_adv', reuse=False): 
            framework = Framework(is_training=True)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
            logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)

        # Add perturbation
        loss = framework.classifier.softmax_cross_entropy(logit)
        new_word_embedding = framework.adversarial(loss, word_embedding)
        new_embedding = framework.embedding.concat_embedding(new_word_embedding, pos_embedding)
        
        # Train
        with tf.variable_scope('pcnn_att_adv', reuse=True): 
            x = framework.encoder.pcnn(new_embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
            logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)
            loss = framework.classifier.softmax_cross_entropy(logit)
            output = framework.classifier.output(logit)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        with tf.variable_scope('pcnn_att_adv', reuse=False): 
            framework = Framework(is_training=False)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
            logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)

        framework.init_test_model(logit)
        framework.load_test_data()
        framework.test()

