'''
File:          wordsData.py
Author:        fis
Created:       Feb 8  2017
Last modified: Feb 10 2017
'''
import pickle
from random import shuffle, randint
import numpy as np

WORDS_DATA_PATH = '../resource/words.pkl'
VALIDATION_SIZE = 0.2
RAND_RANGE = 6


class dataLoader(object):
    def __init__(self):
        with open(WORDS_DATA_PATH, 'rb') as f:
            words = pickle.load(f)
            shuffle(words)
            filteredWords = []
            count = 0
            for word in words:
                if word[1] == 1:
                    filteredWords.append(word)
                elif word[1] == 0 and randint(0, RAND_RANGE) == 0:
                    filteredWords.append(word)
                    count += 1
                else:
                    continue
            self.words = filteredWords
            self.dataSize = len(self.words)
            print('Incorrect / Total: ', count/self.dataSize)

    def getValidationData(self):
        validation = self.words[-round(self.dataSize*VALIDATION_SIZE):]
        parameters, labels = zip(*validation)
        parameters = np.array(parameters, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)
        return (parameters, labels)

    def getTrainingData(self):
        training = self.words[0:round(self.dataSize*(1-VALIDATION_SIZE))]
        parameters, labels = zip(*training)
        parameters = np.array(parameters, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)
        return (parameters, labels)
