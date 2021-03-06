import pandas as pd
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

import theano.sandbox.cuda.basic_ops as sbcuda

n = 50000
seq_len = 100
h = 32
num_tags = 1000
batch_size = 64

gpu = True

print("loading data")
start = time.time()
glove = np.load('glove.npy')

features = np.load('features.npy')[:,40:]
y = np.load('y.npy')

features = torch.from_numpy(features)
y = torch.from_numpy(y).float()
glove = torch.from_numpy(glove)



print(sys.getsizeof(features))
print(sys.getsizeof(y))
print(sys.getsizeof(glove))
#features = features.half()



train_idx = int(np.floor(features.size()[0] * 8 / 10))

train_loader = torch.utils.data.TensorDataset(features[:train_idx ], y[:train_idx ])
test_loader = torch.utils.data.TensorDataset(features[train_idx :], y[train_idx :])


'''
if not gpu:

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True)
else:

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True,pin_memory=True)
'''

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True,pin_memory=True)


print(time.time() - start)
print("creating model")


#model = LSTM_Model(h,glove,num_tags)
model = CNN(glove ,y.size()[1],features.size()[1])
print(sys.getsizeof(model))

params = model.parameters()

opt = optim.Adam(list(params), lr=0.001)

bce = torch.nn.BCELoss()

if gpu:
    model.cuda()
    bce.cuda()


print(model)

def train():

    model.train()

    start = time.time()
    avg_loss = 0
    i = 1
    for data, target in train_loader:

        #data = data.half()
        #target = target.half()

        #if gpu:
            #data = data.cuda()
            #target = target.cuda()

        data, target = Variable(data), Variable(target)

        if gpu:

            data = data.cuda()                               
            target = target.cuda()


        opt.zero_grad()

        y_hat = model.forward(data)

        loss = bce(y_hat, target)

        loss.backward()

        opt.step()

        avg_loss += loss
        i += 1

        if i % 400 == 0:
            print("averge loss: ", (avg_loss / i).data[0], " time elapsed:", time.time() - start)
            print( "CUDA memory: "  ,sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024/1024 )



def test():

    model.eval()

    avg_loss = 0
    avg_fscore = 0

    for data, target in test_loader:

        data, target = Variable(data), Variable(target)

        if gpu:
            target = target.cuda() 
            data = data.cuda()

        y_hat = model.forward(data)

        loss = bce(y_hat, target)

        y_hat = y_hat.cpu()
        target = target.cpu()

        predicted = y_hat.data.numpy().flatten()

        correct = target.data.numpy().flatten()

        score = roc_auc_score(correct, predicted)

        avg_loss += loss
        avg_fscore += score

    print("averge loss: ", (avg_loss / len(test_loader)).data[0], " average f score: ", avg_fscore / len(test_loader))


for i in range(6):
    train()
    test()


