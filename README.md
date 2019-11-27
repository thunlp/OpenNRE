# 中文关系抽取
使用哈工大，BERT-wwm，中文bert，在20w中文人物关系数据上的准确率达到0.97
## 训练结果
=== Epoch 0 train ===
100%|██████████████████████████████████████████████████████████████████| 3094/3094 [40:12<00:00,  1.28it/s, acc=0.773, loss=0.687]
=== Epoch 0 val ===
100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [00:06<00:00,  2.42it/s, acc=0.934]
Best ckpt and saved.
=== Epoch 1 train ===
100%|██████████████████████████████████████████████████████████████████| 3094/3094 [38:17<00:00,  1.35it/s, acc=0.923, loss=0.235]
=== Epoch 1 val ===
100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [00:05<00:00,  2.78it/s, acc=0.972]
Best ckpt and saved.
=== Epoch 2 train ===
100%|██████████████████████████████████████████████████████████████████| 3094/3094 [22:43<00:00,  2.27it/s, acc=0.961, loss=0.121]
=== Epoch 2 val ===
100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [00:05<00:00,  2.71it/s, acc=0.986]
Best ckpt and saved.
Best acc on val set: 0.986000
100%|██████████████████████████████████████████████████████████████████████████████████| 16/16 [00:06<00:00,  2.64it/s, acc=0.986]
Accuracy on test set: 0.986
## 测试结果
model.infer({'text': '场照片事后将发给媒体，避免采访时出现混乱，[3]举行婚礼侯佩岑黄伯俊婚纱照2011年4月17日下午2点，70名亲友见 证下，侯佩', 'h': {'pos': (28, 30)}, 't': {'pos': (31, 33)}})
('夫妻', 0.9995878338813782)

model.infer({'text': '及他们的女儿小苹果与汪峰感情纠葛2004年，葛荟婕在欧洲杯期间录制节目时与汪峰相识并相恋，汪峰那首《我如此爱你', 'h': {'pos': (10, 11)}, 't': {'pos': (22, 24)}})
('情侣', 0.9992896318435669)

model.infer({'text': '14日，彭加木的侄女彭丹凝打通了彭加木儿子彭海的电话，“堂哥已经知道了，他说这些年传得太多，他不相信是真的', 'h': {'pos': (4, 6)}, 't': {'pos': (22, 21)}})
('父母', 0.8954808712005615)

model.infer({'text': '名旦吴菱仙是位列“同治十三绝”的名旦时小福的弟子，算得梅兰芳的开蒙老师，早年曾搭过梅巧玲的四喜班，旧谊', 'h': {'pos': (2, 4)}, 't': {'pos': (27, 29)}})
('师生', 0.996309220790863)
# 使用方式
1.bert模型下载：在./pretrain/下面放置chinese_wwm_pytorch模型
2.数据下载：在./benchmark/people-relation/下执行gen.py,生产中文人物关系数据，具体脚本中有说明。
3.配置环境变量：vim ~/.bash_profile  添加
    # openNRE
    export openNRE=项目位置


-----
以下是原工程内容
# OpenNRE

OpenNRE is an open-source and extensible toolkit that provides a unified framework to implement relation extraction models. This package is designed for the following groups:

* **New to relation extraction**: We have hand-by-hand tutorials and detailed documents that can not only enable you to use relation extraction tools, but also help you better understand the research progress in this field.
* **Developers**: Our easy-to-use interface and high-performance implementation can acclerate your deployment in the real-world applications. Besides, we provide several pretrained models which can be put into production without any training.
* **Researchers**: With our modular design, various task settings and metric tools, you can easily carry out experiments on your own models with only minor modification. We have also provided several most-used benchmarks for different settings of relation extraction.
* **Anyone who need to submit an NLP homework to impress their professors**: With state-of-the-art models, our package can definitely help you stand out among your classmates!

This package is mainly contributed by [Tianyu Gao](https://github.com/gaotianyu1350), [Xu Han](https://github.com/THUCSTHanxu13), [Shulian Cao](https://github.com/ShulinCao), [Lumin Tang](https://github.com/Tsingularity), [Yankai Lin](https://github.com/Mrlyk423), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/)

## What is Relation Extraction

Relation extraction is a natural language processing (NLP) task aiming at extracting relations (e.g., *founder of*) between entities (e.g., **Bill Gates** and **Microsoft**). For example, from the sentence *Bill Gates founded Microsoft*, we can extract the relation triple (**Bill Gates**, *founder of*, **Microsoft**). 

Relation extraction is a crucial technique in automatic knowledge graph construction. By using relation extraction, we can accumulatively extract new relation facts and expand the knowledge graph, which, as a way for machines to understand the human world, has many downstream applications like question answering, recommender system and search engine. 

## How to Cite

A good research work is always accompanied by a thorough and faithful reference. If you use or extend our work, please cite the following paper:

```
    @inproceedings{han2019opennre,
      title={OpenNRE: An Open and Extensible Toolkit for Neural Relation Extraction},
      author={Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong },
      booktitle={Proceedings of EMNLP},
      year={2019}
    }
```

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

Add `OpenNRE` directory to the `PYTHONPATH` environment variable, or open a python session under the `OpenNRE` folder. Then import our package and load pre-trained models.

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

For higher-level usage, you can refer to our [document](https://opennre-docs.readthedocs.io/en/latest/).

## Google Group

If you want to receive our update news or take part in discussions, please join our [Google Group](https://groups.google.com/forum/#!forum/opennre/join)
