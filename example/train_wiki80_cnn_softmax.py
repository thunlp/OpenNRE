import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework

# Some basic settings
root_path = '.'
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/wiki80_cnn_softmax.pth.tar'

# Check data
opennre.download_wiki80(root_path=root_path)
opennre.download_glove(root_path=root_path)
rel2id = json.load(open(os.path.join(root_path, 'benchmark/wiki80/wiki80_rel2id.json')))
wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder
sentence_encoder = opennre.encoder.CNNEncoder(
    token2id=wordi2d,
    max_length=40,
    word_size=50,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.5
)

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=os.path.join(root_path, 'benchmark/wiki80/wiki80_train.txt'),
    val_path=os.path.join(root_path, 'benchmark/wiki80/wiki80_val.txt'),
    test_path=os.path.join(root_path, 'benchmark/wiki80/wiki80_val.txt'),
    model=model,
    ckpt=ckpt,
    batch_size=32,
    max_epoch=2,
    lr=0.1,
    weight_decay=1e-5,
    opt='sgd'
)

# Train the model
framework.train_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('Accuracy on test set: {}'.format(result['acc']))
