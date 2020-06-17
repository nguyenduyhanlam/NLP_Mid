# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:39:00 2020

@author: lam.nguyen
"""
import nltk
nltk.download('brown')
from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing
import re

# Normalize the string (marks and words are seperated, lower the word,...)
def normalizeString(s):
    # Seperate words and marks by adding spaces between them
    marks = '[.!?,-${}()\":\\;/]'
    r = "(["+"\\".join(marks)+"])"
    s = re.sub(r, r" \1 ", s)
    # replace continuous spaces with a single space
    s = re.sub(r"\s+", r" ", s).strip()

    return s.lower()

alpha = 3/4
class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count words in corpus
        self.word2prob = {}

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def generateProb(self):
        for item in self.word2count.items():
            self.word2prob[item[0]] = (item[1] / sum(self.word2count.values())) ** alpha

fileName = 'train.txt'
def read_data(file):
    vocabulary = Vocabulary()
    traindata = []
    sents = open(file, 'r', encoding="utf8").readlines()
    for sent in sents:
      sent = normalizeString(sent)
      vocabulary.addSentence(sent)
      traindata.append(sent.split())
    vocabulary.generateProb()
    return traindata, vocabulary

train, vocabulary = read_data(fileName)
print(train[:2])
print(len(train))

window_size = 5
