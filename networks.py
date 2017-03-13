
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class LSTM_Model(nn.Module):

    def __init__(self,h,glove,num_out):
        super(LSTM_Model, self).__init__()

        self.h = h

        self.embed = nn.Embedding(glove.shape[0], glove.shape[1], padding_idx=0 )
        self.embed.weight = nn.Parameter(torch.from_numpy(glove) )

        self.lstm = nn.LSTM(glove.shape[1], h, 1, batch_first=True)

        self.output_layer = nn.Linear(h, num_out)


    def forward(self,x):

        h0 = Variable(torch.zeros(1, x.size()[0], self.h))
        c0 = Variable(torch.zeros(1, x.size()[0], self.h))

        E = self.embed(x)

        z = self.lstm(E, (h0, c0))[0][:, -1, :]

        y_hat = F.sigmoid(self.output_layer(z))

        return y_hat


class CNN(nn.Module):

    def __init__(self,glove,num_out,seq_len):
        super(CNN, self).__init__()

        self.seq_len = seq_len

        self.embed = nn.Embedding(glove.shape[0], glove.shape[1], padding_idx=0)
        #self.embed.weight = nn.Parameter(torch.from_numpy(glove).float() , requires_grad=False)

        self.conv1 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(2)

        self.flat_dim = self.get_flat_dim()

        self.output_layer = nn.Linear(self.flat_dim, num_out)

    def get_flat_dim(self):

        x = Variable(torch.ones(32,self.seq_len)).long()

        E = self.embed(x)

        E = E.transpose(1, 2).contiguous()

        h = self.pool1(self.drop1(F.relu(self.conv1(E))))
        h = self.pool2(self.drop2(F.relu(self.conv2(h))))
        h = self.pool3(self.drop3(F.relu(self.conv3(h))))

        return(h.size()[1] * h.size()[2])


    def forward(self,x):

        E = self.embed(x)

        E = E.transpose(1, 2).contiguous()

        h = self.pool1(self.drop1(F.relu(self.conv1(E))))
        h = self.pool2(self.drop2(F.relu(self.conv2(h))))
        h = self.pool3(self.drop3(F.relu(self.conv3(h))))

        h = h.view(-1,self.flat_dim)

        return F.sigmoid(self.output_layer(h))
