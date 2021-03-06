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



def tag_freq(tags):
    indiv_tf = {}

    for tag in tags:

        for t in tag:

            if t in indiv_tf:
                indiv_tf[t] += 1
            else:
                indiv_tf[t] = 1

    tf = sorted(list(indiv_tf.items()), key=lambda x: -x[1])

    return tf


def word_freq(lyrics, n):
    vect = CountVectorizer(max_features=n, stop_words='english')

    bag = vect.fit_transform(lyrics).toarray()

    freq = list(bag.sum(axis=0))

    word_freq = [(word, f) for word, f in zip(list(vect.vocabulary_.keys()), freq)]

    word_freq = sorted(word_freq, key=lambda x: -x[1])

    return word_freq


def clean_tags(raw):
    if raw == '':
        return []

    tags = raw[1:-2].split("]")
    tags = [tag.split("'")[1] for tag in tags]
    return tags


def tag2index(tags, tag_map):
    indices = []

    for tag in tags:

        x = []
        for e in tag:

            if e in tag_map:
                x.append(tag_map[e])

        indices.append(x)
    return indices


def sent2seq(text, key, tok, l):
    words = tok(text)

    unknown = len(key.keys()) + 1

    seq = []
    for word in words:
        if word in key:
            seq.append(key[word] + 1)
        else:
            seq.append(unknown)

    if len(seq) > l:
        return seq[:l]
    else:
        padding = [0 for i in range(l - len(seq))]

        return (padding + seq)

    return seq


def load_glove(vocab):
    embedding_mat = [np.zeros(200)]
    new_vocab = {}

    count = 0

    with open('/home/zz1409/glove.6B.200d.txt') as f:

        for i, line in enumerate(f):

            s = line.split()

            if s[0] in vocab:
                embedding_mat.append(np.asarray(s[1:]))
                new_vocab[s[0]] = count
                count += 1

                if len(list(new_vocab.keys())) == len(list(vocab.keys())):
                    return new_vocab, np.array(embedding_mat)

    embedding_mat.append(np.random.randn(200))

    return new_vocab, np.array(embedding_mat).astype(np.float32())
