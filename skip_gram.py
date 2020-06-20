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
import numpy as np

# Normalize the string (marks and words are seperated, lower the word,...)
def normalizeString(s):
    # Seperate words and marks by adding spaces between them
    marks = '[.!?,-${}()\":\\;/]'
    r = "(["+"\\".join(marks)+"])"
    s = re.sub(r, r" \1 ", s)
    # replace continuous spaces with a single space
    s = re.sub(r"\s+", r" ", s).strip()

    return s.lower()

alpha = 1
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

fileName = 'dataset.txt'
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

data, vocabulary = read_data(fileName)
print(data[:2])
print(len(data))

window_size = 5
K = 10
print(sum(vocabulary.word2prob.values()))
neg_words = np.random.choice(list(vocabulary.word2prob.keys()), size=K, p=list(vocabulary.word2prob.values()))

def CreateContextWindow(sentence, index, windowSize):
    
    left_side = index - windowSize
    right_side = index + windowSize
    
    if left_side < 0:
        left_side = 0
    if right_side > len(sentence) - 1:
        right_side = len(sentence) - 1
        
    contextWindow = []
    for i in range(left_side, right_side + 1):
        contextWindow.append(sentence[i])
        
    return contextWindow

def CreateInput(centerWord, k=K):
    input_vector = np.zeros(vocabulary.n_words + k)
    idx = vocabulary.word2index[centerWord]
    input_vector[idx] = 1
    return np.reshape(np.asarray(input_vector), (vocabulary.n_words + k, 1))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

w_input = np.random.rand(vocabulary.n_words + K, window_size * 2 + 1)
w_output = np.random.rand(vocabulary.n_words + K, window_size * 2 + 1)

learning_rate = 0.001
for sentence in data:
    for index,word in enumerate(sentence):
        
        cw = CreateContextWindow(sentence, index, window_size)
        
        for i,c in enumerate(cw):
            
            # if c (c_pos) is word (center_word) -> continue
            if c == word:
                continue
            
            # Get neg_samplings
            wneg = np.random.choice(list(vocabulary.word2prob.keys()), size=K, p=list(vocabulary.word2prob.values()))
            
            # Forward
            input_set = CreateInput(word)
            h =  w_input.T @ input_set
            output_set = w_output @ h
            output_set = sigmoid(output_set)
            
            # Backward
            idx = vocabulary.word2index[c]
            d_out_pos = sigmoid(output_set[idx] @ h.T) - 1
            d_out_neg = sigmoid(output_set[vocabulary.n_words : vocabulary.n_words + K] @ h.T)
            
            dw_output_pos = d_out_pos * h
            dw_output_neg = d_out_neg @ h
            
            dh_pos = d_out_pos * w_output[idx]
            d_out_neg = np.reshape(d_out_neg, (K, 1))
            dh_neg = w_output[vocabulary.n_words : vocabulary.n_words + K].T @ d_out_neg
            
            # Update
            w_output[idx] = w_output[idx] - learning_rate * dw_output_pos
            w_output[vocabulary.n_words : vocabulary.n_words + K] = w_output[vocabulary.n_words : vocabulary.n_words + K] - learning_rate * dw_output_neg
            
            w_input[idx] = w_input[idx] - learning_rate * dh_pos
            w_input[vocabulary.n_words : vocabulary.n_words + K] = w_input[vocabulary.n_words : vocabulary.n_words + K] - learning_rate * dh_neg