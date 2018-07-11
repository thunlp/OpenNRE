from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pcnn_ave_adv(is_training):
    if is_training:
        with tf.variable_scope('pcnn_ave_adv', reuse=False): 
            framework = Framework(is_training=True)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
            x = framework.selector.average(x)

        # Add perturbation
        loss = framework.classifier.softmax_cross_entropy(x)
        new_word_embedding = framework.adversarial(loss, word_embedding)
        new_embedding = framework.embedding.concat_embedding(new_word_embedding, pos_embedding)
        
        # Train
        with tf.variable_scope('pcnn_ave_adv', reuse=True): 
            x = framework.encoder.pcnn(new_embedding, activation=tf.nn.relu)
            x = framework.selector.average(x)
            loss = framework.classifier.softmax_cross_entropy(x)
            output = framework.classifier.output(x)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        with tf.variable_scope('pcnn_ave_adv', reuse=False): 
            framework = Framework(is_training=False)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
            x = framework.selector.average(x)

        framework.init_test_model(tf.nn.softmax(x))
        framework.load_test_data()
        framework.test()

