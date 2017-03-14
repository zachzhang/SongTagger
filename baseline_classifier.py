import pandas as pd
import numpy as np
from utils import *
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import hamming_loss

n = 10000
num_tags = 1000

#load all lyric data into pandas dataframe
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
X = vect.fit_transform(df['lyrics'])
vocab = vect.vocabulary_
tok = vect.build_analyzer()


model = OneVsRestClassifier(MultinomialNB()).fit(X,y)

y_hat = model.predict_proba(X)

correct = y.flatten()
predicted = y_hat.flatten()

auc = roc_auc_score(correct, predicted)

precision, recall, thresholds = precision_recall_curve(correct, predicted)
f_score = 2* precision * recall / (precision + recall)

i_max = f_score.argmax()
f_max = f_score[i_max]
max_thresh  =  thresholds[i_max]

hamming = hamming_loss(y,y_hat > max_thresh)

print("AUC: " , auc)
print("F Score: " , f_max)
print("Hamming Loss: " , hamming)