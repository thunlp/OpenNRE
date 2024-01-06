# OpenNRE (sub-project of OpenSKL)
 
OpenNRE is a sub-project of OpenSKL, providing an **Open**-source **N**eural **R**elation **E**xtraction toolkit for extracting structured knowledge from plain text, with [ATT](https://aclanthology.org/P16-1200.pdf) as key features to consider relation-associated text information.

## Overview

OpenNRE is an open-source and extensible toolkit that provides a unified framework to implement relation extraction models. This package is designed for the following groups:

* **New to relation extraction**: We have hand-by-hand tutorials and detailed documents that can not only enable you to use relation extraction tools, but also help you better understand the research progress in this field.
* **Developers**: Our easy-to-use interface and high-performance implementation can acclerate your deployment in the real-world applications. Besides, we provide several pretrained models which can be put into production without any training.
* **Researchers**: With our modular design, various task settings and metric tools, you can easily carry out experiments on your own models with only minor modification. We have also provided several most-used benchmarks for different settings of relation extraction.
* **Anyone who need to submit an NLP homework to impress their professors**: With state-of-the-art models, our package can definitely help you stand out among your classmates!

## What is Relation Extraction

Relation extraction is a natural language processing (NLP) task aiming at extracting relations (e.g., *founder of*) between entities (e.g., **Bill Gates** and **Microsoft**). For example, from the sentence *Bill Gates founded Microsoft*, we can extract the relation triple (**Bill Gates**, *founder of*, **Microsoft**). 

Relation extraction is a crucial technique in automatic knowledge graph construction. By using relation extraction, we can accumulatively extract new relation facts and expand the knowledge graph, which, as a way for machines to understand the human world, has many downstream applications like question answering, recommender system and search engine. 

It's our honor to help you better explore relation extraction with our OpenNRE toolkit!

## Papers and Document

If you want to learn more about neural relation extraction, visit another project of ours ([NREPapers](https://github.com/thunlp/NREPapers)).

You can refer to our [document](https://opennre-docs.readthedocs.io/en/latest/) for more details about this project.

## Install 

### Install as A Python Package

We are now working on deploy OpenNRE as a Python package. Coming soon!

### Using Git Repository

Clone the repository from our github page (don't forget to star us!)

```bash
git clone https://github.com/thunlp/OpenNRE.git
```

If it is too slow, you can try
```
git clone https://github.com/thunlp/OpenNRE.git --depth 1
```

Then install all the requirements:

```
pip install -r requirements.txt
```

**Note**: Please choose appropriate PyTorch version based on your machine (related to your CUDA version). For details, refer to https://pytorch.org/. 

Then install the package with 
```
python setup.py install 
```

If you also want to modify the code, run this:
```
python setup.py develop
```

Note that we have excluded all data and pretrain files for fast deployment. You can manually download them by running scripts in the ``benchmark`` and ``pretrain`` folders. For example, if you want to download FewRel dataset, you can run

```bash
bash benchmark/download_fewrel.sh
```

## Easy Start

Make sure you have installed OpenNRE as instructed above. Then import our package and load pre-trained models.

```python
>>> import opennre
>>> model = opennre.get_model('wiki80_cnn_softmax')
```

Note that it may take a few minutes to download checkpoint and data for the first time. Then use `infer` to do sentence-level relation extraction

```python
>>> model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
('father', 0.5108704566955566)
```

You will get the relation result and its confidence score.

If you want to use the model on your GPU, just run 
```python
>>> model = model.cuda()
```
before calling the inference function.

For now, we have the following available models:

* `wiki80_cnn_softmax`: trained on `wiki80` dataset with a CNN encoder.
* `wiki80_bert_softmax`: trained on `wiki80` dataset with a BERT encoder.
* `wiki80_bertentity_softmax`: trained on `wiki80` dataset with a BERT encoder (using entity representation concatenation).
* `tacred_bert_softmax`: trained on `TACRED` dataset with a BERT encoder.
* `tacred_bertentity_softmax`: trained on `TACRED` dataset with a BERT encoder (using entity representation concatenation).

## Datasets 

You can go into the `benchmark` folder and download datasets using our scripts. We also list some of the information about the datasets in [this document](https://opennre-docs.readthedocs.io/en/latest/get_started/benchmark.html#bag-level-relation-extraction). We provide two distantly-supervised datasets with human-annotated test sets, **NYT10m** and **Wiki20m**. Check the [datasets](#datasets) section for details.

## Training

You can train your own models on your own data with OpenNRE. In `example` folder we give example training codes for supervised RE models and bag-level RE models. You can either use our provided datasets or your own datasets. For example, you can use the following script to train a PCNN-ATT bag-level model on the NYT10 dataset with manual test set:
```bash
python example/train_bag_cnn.py \
    --metric auc \
    --dataset nyt10m \
    --batch_size 160 \
    --lr 0.1 \
    --weight_decay 1e-5 \
    --max_epoch 100 \
    --max_length 128 \
    --seed 42 \
    --encoder pcnn \
    --aggr att
```

Or use the following script to train a BERT model on the Wiki80 dataset:
```bash
python example/train_supervised_bert.py \
    --pretrain_path bert-base-uncased \
    --dataset wiki80
```

We provide many options in the example training code and you can check them out for detailed instructions.

## Citation

A good research work is always accompanied by a thorough and faithful reference. If you use or extend our work, please cite the following paper:

```
@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}
```

This package is mainly contributed by [Tianyu Gao](https://github.com/gaotianyu1350), [Xu Han](https://github.com/THUCSTHanxu13), [Shulian Cao](https://github.com/ShulinCao), [Lumin Tang](https://github.com/Tsingularity), [Yankai Lin](https://github.com/Mrlyk423), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/)


******************
## About OpenSKL
OpenSKL project aims to harness the power of both structured knowledge and natural languages via representation learning. All sub-projects of OpenSKL, under the categories of **Algorithm**, **Resource** and **Application**, are as follows.

- **Algorithm**: 
  - [OpenKE](https://www.github.com/thunlp/OpenKE)
    - An effective and efficient toolkit for representing structured knowledge in large-scale knowledge graphs as embeddings, with <a href="https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350"> TransR</a> and  <a href="https://aclanthology.org/D15-1082.pdf">PTransE</a> as key features to handle complex relations and relational paths.
    - This toolkit also includes three repositories:
       - [KB2E](https://www.github.com/thunlp/KB2E)
       - [TensorFlow-Transx](https://www.github.com/thunlp/TensorFlow-Transx)
       - [Fast-TransX](https://www.github.com/thunlp/Fast-TransX)
  - [ERNIE](https://github.com/thunlp/ERNIE)
    - An effective and efficient toolkit for augmenting pre-trained language models with knowledge graph representations.
  - [OpenNE](https://www.github.com/thunlp/OpenNE)
    - An effective and efficient toolkit for representing nodes in large-scale graphs as embeddings, with [TADW](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) as key features to incorporate text attributes of nodes.
  - [OpenNRE](https://www.github.com/thunlp/OpenNRE)
    - An effective and efficient toolkit for implementing neural networks for extracting structured knowledge from text, with [ATT](https://aclanthology.org/P16-1200.pdf) as key features to consider relation-associated text information.
    - This toolkit also includes two repositories:
      - [JointNRE](https://www.github.com/thunlp/JointNRE)
      - [NRE](https://github.com/thunlp/NRE)
- **Resource**:
  - The embeddings of large-scale knowledge graphs pre-trained by OpenKE, covering three typical large-scale knowledge graphs: Wikidata, Freebase, and XLORE. The embeddings are free to use under the [MIT license](https://opensource.org/license/mit/), and please click the following link to submit [download requests](http://139.129.163.161/download/wikidata).
  - OpenKE-Wikidata
    - Wikidata is a free and collaborative database, collecting structured data to provide support for Wikipedia. The original Wikidata contains 20,982,733 entities, 594 relations and 68,904,773 triplets. In particular, Wikidata-5M is the core subgraph of Wikidata, containing  5,040,986 high-frequency entities from Wikidata with their corresponding 927 relations and 24,267,796 triplets.
    - TransE version: Knowledge embeddings of Wikidata pre-trained by OpenKE. 
    - [TransR version](https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/transr.npy) of Wikidata-5M: Knowledge embeddings of Wikidata-5M pre-trained by OpenKE for the project [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin).
  - OpenKE-Freebase
    - Freebase was a large collaborative knowledge base consisting of data composed mainly by its community members. It was an online collection of structured data harvested from many sources. Freebase contains 86,054,151 entities, 14,824 relations and 338,586,276 triplets.
    - TransE version: Knowledge embeddings of Freebase pre-trained by OpenKE. 
  - OpenKE-XLORE
    - XLORE is one of the most popular Chinese knowledge graphs developed by THUKEG. XLORE contains 10,572,209 entities, 138,581 relations and 35,954,249 triplets.
    - TransE version: Knowledge embeddings of XLORE pre-trained by OpenKE.
- **Application**:   
    - [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin)
      - An effective and efficient toolkit of plug-and-play knowledge injection for pre-trained language models. Knowledge-Plugin is general for all kinds of knowledge graph embeddings mentioned above. In the toolkit, we provide the example of plugging OpenKE-Wikidata embeddings into BERT.
