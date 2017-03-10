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


print("loading data")
start = time.time()
glove = np.load('glove.npy')
features = np.load('features.npy')
y = np.load('y.npy')

features = torch.from_numpy(features)
y = torch.from_numpy(y)

train_idx = features.size()[0] * 8 / 10

train_loader = torch.utils.data.TensorDataset(features[:train_idx ], y[:train_idx ])
test_loader = torch.utils.data.TensorDataset(features[train_idx :], y[train_idx :])

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True,pin_memory=True)

print(time.time() - start)
print("creating model")


# Create the Model
embed = nn.Embedding(glove.shape[0], 50, padding_idx=0)
embed.weight = nn.Parameter(torch.from_numpy(glove))

model = nn.LSTM(50, h, 1, batch_first=True)

output_layer = nn.Linear(h, num_tags)

#embed.cuda()
#model.cuda()
#output_layer.cuda()


params = list(model.parameters()) + list(embed.parameters()) + list(output_layer.parameters())

#opt = optim.Adam(params, lr=0.001)
opt = optim.SGD(params, lr=0.001)

bce = torch.nn.BCELoss()


def train():
    start = time.time()
    avg_loss = 0
    i = 0
    for data, target in train_loader:

        target= target.float()

        #data = data.cuda()
        #target = target.cuda()

        data, target = Variable(data), Variable(target)

        h0 = Variable(torch.zeros(1, data.size()[0], h))
        c0 = Variable(torch.zeros(1, data.size()[0], h))

        opt.zero_grad()

        E = embed(data)

        z = model(E, (h0, c0))[0][:,-1,:]

        y_hat = F.sigmoid(output_layer(z))

        #z, _ = model(E, (h0, c0))

        #y_hat = F.sigmoid(output_layer(z[:, -1, :]))

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

