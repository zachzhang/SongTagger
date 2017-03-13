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
from networks import *

n = 10000
seq_len = 30
h = 32
num_tags = 1000
batch_size = 64


print("loading data")
start = time.time()
glove = np.load('glove.npy')

#features = np.load('features.npy')[0:1000]
#y = np.load('y.npy')[0:1000]

features = np.load('features.npy')
y = np.load('y.npy')

features = torch.from_numpy(features)
y = torch.from_numpy(y)

train_idx = int(np.floor(features.size()[0] * 8 / 10))

print(train_idx)

train_loader = torch.utils.data.TensorDataset(features[:train_idx ], y[:train_idx ])
test_loader = torch.utils.data.TensorDataset(features[train_idx :], y[train_idx :])

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True)

#train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True,pin_memory=True)
#test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True,pin_memory=True)


print(time.time() - start)
print("creating model")


#model = LSTM_Model(h,glove,num_tags)
model = CNN(glove,num_tags,seq_len)

params = model.parameters()

#print(len(list(params)))

opt = optim.Adam(list(model.conv1.parameters()), lr=0.001)
#opt = optim.Adam(list(model.output_layer.parameters()), lr=0.001)

bce = torch.nn.BCELoss()


def train():

    model.train()

    start = time.time()
    avg_loss = 0
    i = 1
    for data, target in train_loader:

        target= target.float()

        #data = data.cuda()
        #target = target.cuda()

        data, target = Variable(data), Variable(target)

        opt.zero_grad()

        #data = Variable(torch.ones(data.size()[0],50,30))

        y_hat = model.forward(data)

        loss = bce(y_hat, target)

        loss.backward()

        opt.step()

        avg_loss += loss
        i += 1

        if i % 20 == 0:
            print("averge loss: ", (avg_loss / i).data[0], " time elapsed:", time.time() - start)



def test():

    model.eval()

    avg_loss = 0
    avg_fscore = 0

    for data, target in test_loader:
        
        #target = target.float().cuda()
        #data = data.cuda()

        data, target = Variable(data), Variable(target)

        opt.zero_grad()

        y_hat = model.forward(data)

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

