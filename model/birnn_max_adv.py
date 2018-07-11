from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def birnn_max_adv(is_training):
    if is_training:
        with tf.variable_scope('birnn_max_adv', reuse=False): 
            framework = Framework(is_training=True)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.birnn(embedding)
            x = framework.selector.maximum(x)

        # Add perturbation 
        loss = framework.classifier.softmax_cross_entropy(x)
        new_word_embedding = framework.adversarial(loss, word_embedding)
        new_embedding = framework.embedding.concat_embedding(new_word_embedding, pos_embedding)
        
        # Train
        with tf.variable_scope('birnn_max_adv', reuse=True): 
            x = framework.encoder.birnn(new_embedding)
            x = framework.selector.maximum(x)
            loss = framework.classifier.softmax_cross_entropy(x)
            output = framework.classifier.output(x)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        with tf.variable_scope('birnn_max_adv', reuse=False): 
            framework = Framework(is_training=False)
            word_embedding = framework.embedding.word_embedding()
            pos_embedding = framework.embedding.pos_embedding()
            embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
            x = framework.encoder.birnn(embedding)
            x = framework.selector.maximum(x)

        framework.init_test_model(tf.nn.softmax(x))
        framework.load_test_data()
        framework.test()

