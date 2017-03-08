import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import torchvision
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import time
import sys
from utils import *

n = 10000
seq_len = 100
h = 128
num_tags = 1000

# load all lyric data into pandas dataframe
df = pd.read_csv('lyric_data.csv', index_col=0)

# Sometimes the API returns an error message rather than actual lyrics. This removes it
bad_song = df['lyrics'].value_counts().index[0]
df[df['lyrics'] == bad_song] = ''

# only take the ones that we have data for
df.fillna('', inplace=True)
df = df[df['lyrics'] != '']

# List of list of tags for each example
tags = [clean_tags(raw) for raw in list(df['tags'])]

# list of tuples of (tag, frequency) in desending order
tf = tag_freq(tags)

# Choose which tags to restrict model too
important_tags = [x[0] for x in tf[0:num_tags]]
important_tags = dict(zip(important_tags, range(len(important_tags))))

# maps each of the tags int 'tags' to an int index
indices = tag2index(tags, important_tags)

# Convert indices to binary vectors of tags
y = np.zeros((len(indices), num_tags))
for i, tags in enumerate(indices):
    for tag in tags:
        y[i, tag] = 1

# Build vocabulary and tokenizer
vect = CountVectorizer(max_features=n, stop_words='english')
vect.fit(df['lyrics'])
vocab = vect.vocabulary_
tok = vect.build_analyzer()

# Load glove vectors for word embedding
vocab, glove = load_glove(vocab)

# Convert text to sequence input
features = df['lyrics'].apply(lambda x: sent2seq(x, vocab, tok, seq_len))
features = np.array(list(features))

# Train loader
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features[:150000]), torch.from_numpy(y[:150000]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Test Loader
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(features[150000:]), torch.from_numpy(y[150000:]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Create the Model
embed = nn.Embedding(glove.shape[0], 50, padding_idx=0)
embed.weight = nn.Parameter(torch.from_numpy(glove))


model = nn.LSTM(50, h, 1, batch_first=True)

output_layer = nn.Linear(h, num_tags)

params = list(model.parameters()) + list(embed.parameters()) + list(output_layer.parameters())

opt = optim.Adam(params, lr=0.001)
bce = torch.nn.BCELoss()


def train():
    start = time.time()
    avg_loss = 0
    i = 0
    for data, target in train_loader:

        data, target = Variable(data), Variable(target.float())
        print(data.max())

        h0 = Variable(torch.zeros(1, data.size()[0], h))
        c0 = Variable(torch.zeros(1, data.size()[0], h))

        opt.zero_grad()

        E = embed(data)

        h, _ = model(E, (h0, c0))

        y_hat = F.sigmoid(output_layer(h[:, -1, :]))

        loss = bce(y_hat, target)

        loss.backward()

        opt.step()

        avg_loss += loss
        i += 1

        if i % 200 == 0:
            print("averge loss: ", (avg_loss / i).data[0], " time elapsed:", time.time() - start)



def test():
    avg_loss = 0
    avg_fscore = 0

    for data, target in test_loader:
        data, target = Variable(data), Variable(target.float())

        h0 = Variable(torch.zeros(1, data.size()[0], h))
        c0 = Variable(torch.zeros(1, data.size()[0], h))

        opt.zero_grad()

        E = embed(data)

        h, _ = model(E, (h0, c0))

        y_hat = F.sigmoid(output_layer(h[:, -1, :]))

        loss = bce(y_hat, target)

        predicted = y_hat.data.numpy().flatten()
        correct = target.data.numpy().flatten()

        score = roc_auc_score(correct, predicted)

        avg_loss += loss
        avg_fscore += score

    print("averge loss: ", (avg_loss / len(test_loader)).data[0], " average f score: ", avg_fscore / len(test_loader))


for i in range(10):
    train()
    test()