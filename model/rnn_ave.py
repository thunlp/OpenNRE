from framework import Framework
import tensorflow as tf

def rnn_ave(is_training):
    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)

    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding(simple_pos=True)
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    x = framework.encoder.rnn(embedding)
    x = framework.selector.average(x)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(x)
        output = framework.classifier.output(x)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(tf.nn.softmax(x))
        framework.load_test_data()
        framework.test()

