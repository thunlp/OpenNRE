# OpenNRE

An open-source framework for neural relation extraction.

Contributed by [Tianyu Gao](https://github.com/gaotianyu1350), [Xu Han](https://github.com/THUCSTHanxu13), [Lumin Tang](https://github.com/Tsingularity), [Yankai Lin](https://github.com/Mrlyk423), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/)

## Overview

It is a TensorFlow-based framwork for easily building relation extraction models. We divide the pipeline of relation extraction into four parts, which are embedding, encoder, selector and classifier. For each part we have implemented several methods.

* Embedding
  * Word embedding
  * Position embedding
  * Concatenation method
* Encoder
  * PCNN
  * CNN
  * RNN
  * BiRNN
* Selector
  * Attention
  * Maximum
  * Average
* Classifier
  * Softmax loss function
  * Output
  
All those methods could be combined freely. 

We also provide fast training and testing codes. You could change hyper-parameters or appoint model architectures by using Python arguments. A plotting method is also in the package.

Advesarial training method is implemented following [Wu et al. (2017)](https://people.eecs.berkeley.edu/~russell/papers/emnlp17-relation.pdf). Since it's a general training method, you could adapt it to any models with simply adding a few lines of code.

This project is under MIT license.

## Installation

1. Install TensorFlow
2. Clone the OpenNRE repository:
  ```bash
  git clone git@github.com:thunlp/OpenNRE.git
  ```
3. Download NYT dataset from `https://drive.google.com/file/d/1BnyXMJ71jM0kxyJUlXa5MHf-vNjE57i-/view?usp=sharing`
4. Extract dataset to `./origin-data`
```
tar xvf origin_data.tar
```

## Quick Start

### Process Data

```bash
python gen_data.py
```
The processed data will be stored in `./data`

### Train Model
```
python train.py --model_name pcnn_att
```

The arg `model_name` appoints model architecture, and `pcnn_att` is the name of one of our models. All available models are in `./model`. About other arguments please refer to `./train.py`. Once you start training, all checkpoints are stored in `./checkpoint`.

### Test Model
```bash
python test.py --model_name pcnn_att
```

Same usage as training. When finishing testing, the best checkpoint's corresponding pr-curve data will be stored in `./test_result`.

### Plot
```bash
python draw_plot.py pcnn_att
```

The plot will be saved as `./test_result/pr_curve.png`. You could appoint several models in the arguments, like `python draw_plot.py pcnn_att pcnn_max pcnn_ave`, as long as there are these models' results in `./test_result`.

## Build Your Own Model

Not only could you train and test existing models in our package, you could also build your own model or add methods to the four basic modules. When adding a new model, you could create a python file in `./model` having the same name as the model and implement it like following:

```python
from framework import Framework
import tensorflow as tf

def your_new_model(is_training):
    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)

    # Word Embedding
    word_embedding = framework.embedding.word_embedding()
    # Position Embedding. Set simple_pos=True to use simple pos embedding
    pos_embedding = framework.embedding.pos_embedding()
    # Concat two embeddings
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    
    # PCNN. Appoint activation to whatever activation function you want to use.
    # There are three more encoders:
    #     framework.encoder.cnn(x, activation=tf.nn.relu)
    #     framework.encoder.rnn(x, cell_name='lstm')
    #     framework.encoder.birnn(x, cell_name='lstm')
    x = framework.encoder.pcnn(embedding, activation=tf.nn.relu)
    
    # Selective attention. Setting parameter dropout_before=True means using dropout before attention. 
    # There are three more selecting method
    #     framework.selector.maximum(x, dropout_before=True)
    #     framework.selector.average(x, dropout_before=True)
    #     framework.selector.no_bag(x)
    x = framework.selector.attention(x)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(x)
        output = framework.classifier.output(x)
        # Set optimizer to whatever optimizer you want to use
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(tf.nn.softmax(x))
        framework.load_test_data()
        framework.test()
```

After creating model's python file, you need to add the model to `./train.py` and `./test.py` as following:

```python

# other code ...

def main():
    from model.your_new_model import your_new_model

# other code ...

```

Then you can train, test and plot!

As for using adversarial training, please refer to `./model/pcnn_att_adv.py` for more details.
