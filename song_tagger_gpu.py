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
seq_len = 30
h = 128
num_tags = 1000
batch_size = 32


dtype = torch.cuda.FloatTensor

# load all lyric data into pandas dataframe
df = pd.read_csv('lyric_data.csv', index_col=0)#.iloc[0:200000]
#df = pd.read_csv('lyric_data_small.csv', index_col=0)




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

np.save('features.npy',features)
np.save('y.npy',y)
np.save('glove.npy',glove)


quit()

features = torch.from_numpy(features)
y = torch.from_numpy(y)

train_idx = features.size()[0] * 8 / 10

print(type(features))

# Train loader
train_dataset = torch.utils.data.TensorDataset(features[:train_idx ], y[:train_idx ])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)

# Test Loader
test_dataset = torch.utils.data.TensorDataset(features[train_idx :], y[train_idx :])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)

pickle.dump(train_dataset, open('train_loader.p','wb'))
pickle.dump(test_dataset, open('test_loader.p','wb'))
pickle.dump(glove,open('glove.p','wb'))

print('done')

# Create the Model
embed = nn.Embedding(glove.shape[0], 50, padding_idx=0)
embed.weight = nn.Parameter(torch.from_numpy(glove))

model = nn.LSTM(50, h, 1, batch_first=True)

output_layer = nn.Linear(h, num_tags)

embed.cuda()
model.cuda()
output_layer.cuda()

params = list(model.parameters()) + list(embed.parameters()) + list(output_layer.parameters())

opt = optim.Adam(params, lr=0.001)
bce = torch.nn.BCELoss().cuda()


def train():
    start = time.time()
    avg_loss = 0
    i = 0
    for data, target in train_loader:

        target= target.float()
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data), Variable(target)

        h0 = Variable(torch.zeros(1, data.size()[0], h).cuda())
        c0 = Variable(torch.zeros(1, data.size()[0], h).cuda())

        opt.zero_grad()

        E = embed(data)

        z, _ = model(E, (h0, c0))

        y_hat = F.sigmoid(output_layer(z[:, -1, :]))

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
        
        target = target.float().cuda()
        data = data.cuda()

        data, target = Variable(data), Variable(target)

        h0 = Variable(torch.zeros(1, data.size()[0], h).cuda())
        c0 = Variable(torch.zeros(1, data.size()[0], h).cuda())

        opt.zero_grad()

        E = embed(data)

        z, _ = model(E, (h0, c0))

        y_hat = F.sigmoid(output_layer(z[:, -1, :]))

        loss = bce(y_hat, target)

        #predicted = y_hat.data.numpy().flatten()
        #correct = target.data.numpy().flatten()

        #score = roc_auc_score(correct, predicted)

        avg_loss += loss
        #avg_fscore += score

    print("averge loss: ", (avg_loss / len(test_loader)).data[0], " average f score: ", avg_fscore / len(test_loader))


for i in range(10):
    train()
    #test()

