# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 01:52:43 2020

@author: User
"""

import nltk
nltk.download('brown')
from nltk.corpus import brown
from gensim.models import Word2Vec
import gensim
import multiprocessing
import re
import numpy as np
from sklearn import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
import gensim.models.keyedvectors as word2vec
import scipy
from scipy.special import softmax

# Normalize the string (marks and words are seperated, lower the word,...)
def normalizeString(s):
    # Seperate words and marks by adding spaces between them
    marks = '[.!?,-${}()\":\\;/]'
    r = "(["+"\\".join(marks)+"])"
    s = re.sub(r, r" \1 ", s)
    # replace continuous spaces with a single space
    s = re.sub(r"\s+", r" ", s).strip()

    return s.lower()

fileName = "dataset.txt"
X_dataset = []
datasetW2v = []
with open(fileName, encoding='utf-8') as f:
  for line in f:
    sp = normalizeString(line)
    X_dataset.append(sp)
    sp = line.split()
    datasetW2v.append(sp)
#print(datasetW2v)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_dataset)
vdict = tokenizer.word_index
print(len(vdict))
print(vdict['cha'])
#print(vdict.items())

sentences = brown.sents()
hidden_unit = 500
#w2v = Word2Vec(datasetW2v, size=hidden_unit, window=5, min_count=10, negative=4, iter=10, workers=multiprocessing.cpu_count())
w2v = gensim.models.KeyedVectors.load_word2vec_format('wiki.vi.model.bin', binary=True)
#w2v = word2vec.KeyedVectors.load_word2vec_format('baomoi.model.bin', binary=True)
wv = w2v.wv
print(w2v['cha'].shape)

sen = "ngu như một con"
h = np.array([wv.word_vec(t) for t in sen.split()])  # 1. Obtain all the word vectors of context words
h = np.mean(h, axis=0)  # 2. Average them to find out the hidden layer vector h of size Nx1
z = np.matmul(wv.syn0, h)  # 3&4. Multiply syn1 by h, the resulting vector will be z with size Vx1
y = softmax(z)  # 5. Compute the probability vector y = softmax(z)
pred_word = w2v.index2word[y.argmax()]  # 6. Return the word
