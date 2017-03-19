'''
File:          dataLoader.py
Author:        fis
Last modified: 25 JAN 2017
'''

import pickle


def wordsSegmentationData():
    wordsPath = '../resource/base_words.pkl'
    with open(wordsPath, 'rb') as f:
        words = pickle.load(f)
    images, labels = words
    return images, labels
