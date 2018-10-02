# OpenNRE

An open-source framework for neural relation extraction.

Contributed by [Tianyu Gao](https://github.com/gaotianyu1350), [Xu Han](https://github.com/THUCSTHanxu13), [Shulian Cao](https://github.com/ShulinCao), [Lumin Tang](https://github.com/Tsingularity), [Yankai Lin](https://github.com/Mrlyk423), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/)

**BIG UPDATE**: The project has been completely reconstructed and is faster, more extendable and the codes are easier to read and use now. If you need get to the old version, please refer to branch [old_version](https://github.com/thunlp/OpenNRE/tree/old_version).  

New features:

- JSON data support.
- Multi GPU training.
- Validating while training.

## Overview

It is a TensorFlow-based framwork for easily building relation extraction (RE) models. We divide the pipeline of relation extraction into four parts, which are embedding, encoder, selector (for distant supervision) and classifier. For each part we have implemented several methods.

* Embedding
  * Word embedding
  * Position embedding
* Encoder
  * PCNN
  * CNN
  * RNN
  * Bidirection RNN
* Selector
  * Attention
  * Maximum
  * Average
* Classifier
  * Softmax Loss Function
  * Output
  
All those methods could be combined freely. 

We also provide training and testing framework for sentence-level RE and bag-level RE. A plotting tool is also in the package.

This project is under MIT license.

## Requirements

- Python (>=2.7)
- Numpy (>=1.13.3)
- TensorFlow (>=1.4.1)
    - CUDA (>=8.0) if you are using gpu
- Matplotlib (>=2.0.0)
- scikit-learn (>=0.18)

## Data Format

For training and testing, you should provide four `JSON` files including training data, testing data, word embedding data and relation-ID mapping data. 

### Training Data & Testing Data

Training data file and testing data file, containing sentences and their corresponding entity pairs and relations, should be in the following format

```
[
    {
        'sentence': 'Bill Gates is the founder of Microsoft .',
        'head': {'word': 'Bill Gates', ...(other information)},
        'tail': {'word': 'Microsoft', ...(other information)},
        'relation': 'founder'
    },
    ...
]
```

**IMPORTANT**: In the sentence part, words and punctuations should be separated by blank spaces.

### Word Embedding Data

Word embedding data is used to initialize word embedding in the networks, and should be in the following format

```
[
    {'word': 'the', 'vec': [0.418, 0.24968, ...]},
    {'word': ',', 'vec': [0.013441, 0.23682, ...]},
    ...
]
```

### Relation-ID Mapping Data

This file indicates corresponding IDs for relations to make sure during each training and testing period, the same ID means the same relation. Its format is as follows

```
{
    'NA': 0,
    'relation_1': 1,
    'relation_2': 2,
    ...
}
```

**IMPORTANT**: Make sure the ID of `NA` is always 0.

## Provided Data

### NYT10 Dataset

NYT10 is a distantly supervised dataset originally released by the paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text.". Here is the download [link](http://iesl.cs.umass.edu/riedel/ecml/) for the original data.

We've provided a toolkit to convert the original NYT10 data into JSON format that `OpenNRE` could use. You could download the original data + toolkit from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/11391e48b72749d8b60a/?dl=1). Further instructions are included in the toolkit.

## Installation and Quick Start

1. Install all the required package.

2. Clone the OpenNRE repository:

```bash
git clone https://github.com/thunlp/OpenNRE.git
```

3. Make data folder in the following structure

```
OpenNRE
|-- ... 
|-- data
    |
    |-- {DATASET_NAME_1}
    |   |
    |   |-- train.json
    |   |-- test.json
    |   |-- word_vec.json
    |   |-- rel2id.json
    |
    |-- {DATASET_NAME_2}
    |   |
    |   |-- ...
    |
    |-- ...
```

You could use your own data or download datasets provided above.

4. Run `train_demo.py {DATASET_NAME} {ENCODER_NAME} {SELECTOR_NAME}`. For example, if you want to train model with PCNN as the encoder and attention as the selector on the `nyt` dataset, run the following command

```
python train_demo.py nyt pcnn att
```

Currently `{ENCODER_NAME}` includes `pcnn`, `cnn`, `rnn` and `birnn`, and `{SELECTOR_NAME}` includes `att` (for attention), `max` (for maximum) and `ave` (for average). The model will be named as `{DATASET_NAME}_{ENCODER_NAME}_{SELECTOR_NAME}` automatically.

The checkpoint of the best epoch (each epoch will be validated while training) will be saved in `./checkpoint` and results for plotting precision-recall curve will be saved in `./test_result` by default.

5. Use `draw_plot.py` to check auc, average precision, F1 score and precision-recall curve by the following command

```
python draw_plot.py {MODEL_NAME_1} {MODEL_NAME_2} ...
```

All the results of the models mentioned will be printed and precision-recall curves containing all the models will be saved in `./test_result/pr_curve.png`.

## Reference

1. **Neural Relation Extraction with Selective Attention over Instances.** _Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun._ ACL2016. [paper](http://www.aclweb.org/anthology/P16-1200)

2. **Adversarial Training for Relation Extraction.** _Yi Wu, David Bamman, Stuart Russell._ EMNLP2017. [paper](http://www.aclweb.org/anthology/D17-1187)

3. **A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction.** _Tianyu Liu, Kexiang Wang, Baobao Chang, Zhifang Sui._ EMNLP2017. [paper](http://aclweb.org/anthology/D17-1189)

4. **Reinforcement Learning for Relation Classification from Noisy Data.** _Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu._ AAAI2018. [paper](https://tianjun.me/static/essay_resources/RelationExtraction/Paper/AAAI2018Denoising.pdf)
