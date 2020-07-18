# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:56:37 2020

@author: lam.nguyen
"""
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load document
in_filename = 'the_republic.txt'
doc = load_doc(in_filename)
print(doc[:200])