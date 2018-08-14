from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_att_rl(is_training):
    if is_training:
        framework = Framework(is_training=True)
        
        with tf.variable_scope('policy_agent', reuse=tf.AUTO_REUSE):
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.cnn(embedding, 2, None, activation=tf.nn.relu)
            logit, repre = framework.selector.no_bag(x, num_classes=2)
            policy_agent_loss = framework.classifier.softmax_cross_entropy(logit, 2)
            policy_agent_output = framework.classifier.output(logit)

        word_embedding = framework.embedding.word_embedding()
        pos_embedding = framework.embedding.pos_embedding()
        embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
        x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
        logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)
        loss = framework.classifier.softmax_cross_entropy(logit)
        output = framework.classifier.output(logit)
        
        framework.init_policy_agent(policy_agent_loss, policy_agent_output, optimizer=tf.train.GradientDescentOptimizer)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        
        framework.train(max_epoch=5)
        framework.pretrain_policy_agent(max_epoch=1)
        framework.train_rl()
    else:
        framework = Framework(is_training=False)
        word_embedding = framework.embedding.word_embedding()
        pos_embedding = framework.embedding.pos_embedding()
        embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
        x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
        logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)

        framework.init_test_model(logit)
        framework.load_test_data()
        framework.test()

