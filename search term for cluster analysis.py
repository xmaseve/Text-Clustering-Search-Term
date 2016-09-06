# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 20:29:46 2016

@author: YI
"""

import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

searchterm = pd.read_csv('C:\Users\YI\Desktop\search term.csv')
searchterm.head()
data = searchterm.values.tolist()

def preprocess(data):
    cleantext = []
    for row in range(len(data)):
        letters_only = re.sub("[^a-zA-Z]", " ", data[row][0])
        words=letters_only.split()
        stop=set(stopwords.words('english'))
        meanfulwords=[w for w in words if not w in stop]
        cleantext.append(meanfulwords)
    return cleantext

cleantext=preprocess(data)

def createVocabList(cleantext):
    vocabSet = set([])
    for i in cleantext:
        vocabSet = vocabSet | set(i)
    vocabList = list(vocabSet)
    return vocabList

def bagOfWords2Vec(vocabList, inputset):
    returnVec = [0] * len(vocabList)
    for word in inputset:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
    
def createMatrix(data, vocablist):
    mat = []
    for inputset in data:
        mat.append(bagOfWords2Vec(vocablist, inputset))
    return mat

vocabList = createVocabList(cleantext) 
'''
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

data_features = vectorizer.fit_transform(vocabList)
data_features = data_features.toarray()
vocab = vectorizer.get_feature_names()
print vocab

# Sum up the counts of each vocabulary word
dist = np.sum(data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag
'''