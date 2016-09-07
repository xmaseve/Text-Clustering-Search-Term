# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 20:29:46 2016

@author: YI
"""

import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.cluster import KMeans


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
mat = createMatrix(cleantext,vocabList)

def kmeans(k):
    kmeans_clustering = KMeans(n_clusters = k)
    idx = kmeans_clustering.fit_predict(mat)
    word_centroid_map = dict(zip( vocabList, idx ))
    for cluster in xrange(0,k):
    #
    # Print the cluster number  
        print "\nCluster %d" % cluster
    #
    # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print words



