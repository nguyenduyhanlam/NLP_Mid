# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:38:35 2020

@author: lam.nguyen
"""
import fasttext
model = fasttext.train_unsupervised('dataset.txt', "skipgram")
words = model.words
print(words)
word_vector = model.get_word_vector("mẹ")
print(word_vector)