
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
        self.embed = nn.Embedding(glove.size()[0], glove.size()[1], padding_idx=0 )
        self.embed.weight = nn.Parameter(glove )

        #self.lstm = nn.LSTM(glove.size()[1], h, 1, batch_first=True)
        self.lstm = nn.GRU(glove.size()[1], h, 1, batch_first=True , dropout = .5)

        self.output_layer = nn.Linear(h, num_out,bias=False)

        self.params = list(self.embed.parameters()) + list(self.output_layer.parameters()) + list(self.lstm.parameters())
        #self.params =  list(self.output_layer.parameters()) + list(self.lstm.parameters())


    def forward(self,x):

        h0 = Variable(torch.zeros(1, x.size()[0], self.h))

        E = self.embed(x)
        
        z = self.lstm(E, h0)[0][:, -1, :]

        #z = self.lstm(E, h0)[1]
        #z = z.transpose(0,1).contiguous().view(-1,2*self.h)

        y_hat = F.sigmoid(self.output_layer(z))

        return y_hat


class BiConvGRU(nn.Module):

    def __init__(self, h,conv_feat ,  glove, num_out,bidirectional = False , pooling = False):

        super(BiConvGRU, self).__init__()

        self.use_pool = pooling
        self.bidirectional = bidirectional

        self.h = h
        self.conv_features = conv_feat
        self.embed = nn.Embedding(glove.size()[0], glove.size()[1], padding_idx=0)
        self.embed.weight = nn.Parameter(glove)

        self.conv = nn.Conv1d(in_channels= glove.size()[1], out_channels=self.conv_features, kernel_size=3)

        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.GRU(self.conv_features, h, 1 , bidirectional = bidirectional, batch_first=True, dropout=.3)

        if bidirectional:
            self.output_layer = nn.Linear(h*2, num_out, bias=False)
        else:
            self.output_layer = nn.Linear(h ,  num_out, bias=False)

        self.params = list(self.embed.parameters()) + list(self.output_layer.parameters()) + list(self.lstm.parameters()) + list(self.conv.parameters())


    def forward(self, x):

        E = self.embed(x)
        E = E.transpose(1, 2).contiguous()

        h = F.relu(self.conv(E))

        if self.use_pool:
            h = self.pool(h)

        h = h.transpose(1, 2).contiguous()

        if self.bidirectional:
            h0 = Variable(torch.zeros(2, x.size()[0], self.h))
        else:
            h0 = Variable(torch.zeros(1, x.size()[0], self.h))

        z = self.lstm(h, h0)[1]

        if self.bidirectional:
            z = z.transpose(0,1).contiguous().view(-1,2*self.h)
        else:
            z = z.transpose(0, 1).contiguous().view(-1, self.h)

        y_hat = F.sigmoid(self.output_layer(z))

        return y_hat


class CNN(nn.Module):

    def __init__(self,glove,num_out,seq_len):
        super(CNN, self).__init__()

        self.seq_len = seq_len

        self.embed = nn.Embedding(glove.size()[0], glove.size()[1], padding_idx=0)
        self.embed.weight = nn.Parameter(glove)

        self.conv1 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.drop4 = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)
        self.bn5 = nn.BatchNorm1d(100)

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(2)
        self.pool4 = nn.MaxPool1d(2)


        self.flat_dim = self.get_flat_dim()

        self.output_layer = nn.Linear(self.flat_dim, num_out,bias=False)

        self.params = self.parameters()

    def get_flat_dim(self):

        x = Variable(torch.ones(32,self.seq_len)).long()

        E = self.embed(x)

        E = E.transpose(1, 2).contiguous()

        h = F.relu(self.bn1(self.conv1(E)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool1(h)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.pool2(h)
        h = F.relu(self.bn5(self.conv5(h)))
        h = self.pool3(h)

        print(h.size()[1] , h.size()[2])

        return(h.size()[1] * h.size()[2])

    def forward(self, x):
        E = self.embed(x)

        E = E.transpose(1, 2).contiguous()

        h = F.relu(self.bn1(self.conv1(E)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool1(h)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.pool2(h)
        h = F.relu(self.bn5(self.conv5(h)))
        h = self.pool3(h)

        h = h.view(-1, self.flat_dim)

        return F.sigmoid(self.output_layer(h))

class CNN2(nn.Module):
    def __init__(self, glove, num_out, seq_len):
        super(CNN2, self).__init__()

        self.seq_len = seq_len
        self.NB_FILTER = 256
        self.embed = nn.Embedding(glove.size()[0], glove.size()[1], padding_idx=0)
        self.embed.weight = nn.Parameter(glove)

        self.ngrams = [ 1,2,3,4,5]

        self.conv_layers = []

        for i in self.ngrams:
            conv = nn.Sequential(
                nn.Conv1d(in_channels=glove.size()[1], out_channels=self.NB_FILTER, kernel_size=i),
                nn.BatchNorm1d(self.NB_FILTER),
                nn.ReLU(),
                nn.Dropout(p=.5)
            )

            self.conv_layers.append(conv)

        self.output_layer = nn.Linear( len(self.ngrams) * self.NB_FILTER, num_out, bias=False)

        #self.params = self.parameters()

    def forward(self,x):

        E = self.embed(x)
        E = E.transpose(1, 2).contiguous()

        conv_h = [  conv(E).max(2)[0] for conv in self.conv_layers]

        h = torch.cat(conv_h,2)
        h = h.view( -1,len(self.ngrams) * self.NB_FILTER )

        y_hat = self.output_layer(h)

        return(y_hat)


class GRU_Attention(nn.Module):

    def __init__(self, h, conv_feat, glove, num_out, bidirectional=False):

        super(GRU_Attention, self).__init__()

        self.bidirectional = bidirectional

        self.h = h
        self.conv_features = conv_feat
        self.embed = nn.Embedding(glove.size()[0], glove.size()[1], padding_idx=0)
        self.embed.weight = nn.Parameter(glove)

        self.conv = nn.Conv1d(in_channels=glove.size()[1], out_channels=self.conv_features, kernel_size=3)

        self.lstm = nn.GRU(self.conv_features, h, 1, bidirectional=bidirectional, batch_first=True, dropout=.3)

        self.num_dir = 2 if bidirectional else 1

        self.output_layer = nn.Linear(h * self.num_dir, num_out, bias=False)

        self.U = nn.Conv1d(in_channels=h * self.num_dir, out_channels=h * self.num_dir , kernel_size=1)

        self.u_a = nn.Parameter(torch.randn(1,1,h * self.num_dir))

        self.params = list(self.embed.parameters()) + list(self.output_layer.parameters()) + list(
            self.lstm.parameters()) + list(self.conv.parameters()) + list(self.U.parameters()) + [self.u_a]


    def forward(self, x):

        h0 = Variable(torch.zeros(self.num_dir, x.size()[0], self.h))

        #embedding layer
        E = self.embed(x)
        E = E.transpose(1, 2).contiguous()

        #conv layer
        h = F.relu(self.conv(E))
        h = h.transpose(1, 2).contiguous()

        #Bidir GRU
        z = self.lstm(h, h0)[0].transpose(1, 2).contiguous()

        #soft attention
        u_t = F.tanh(self.U(z))
        a_t = torch.bmm( self.u_a.expand(x.size()[0],1, self.num_dir * self.h), u_t)
        a_t = F.softmax(a_t.squeeze(1)).unsqueeze(2)

        #representation
        s = torch.bmm(z , a_t).squeeze(2)

        #output
        y_hat = F.sigmoid(self.output_layer(s))

        return y_hat


#model = GRU_Attention(h=128, conv_feat=100, glove=torch.randn(10000,100), num_out=100, bidirectional=True)


model = CNN2( glove=torch.randn(10000,200),num_out=100,seq_len = 50)
print(model(Variable(torch.ones(64, 50)).long()).size())

#model = BiConvGRU( h = 256,conv_feat=200, glove = torch.randn(10000,100), num_out = 100,bidirectional = False , pooling = False)


#print(model(Variable(torch.ones(32,20)).long() ).size())
