from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def rnn_att(is_training):
    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)

    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    x = framework.encoder.rnn(embedding, FLAGS.hidden_size, framework.length)
    logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(logit)
        output = framework.classifier.output(logit)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(tf.nn.softmax(logit))
        framework.load_test_data()
        framework.test()

