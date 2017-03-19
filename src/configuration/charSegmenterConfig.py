'''
File:          charSegmenterConfig.py
Author:        fis
Created:       Feb 8  2017
Last modified: Feb 15 2017
'''
import os
EPOCH = 1000
BATCH_SIZE = 256

WEIGHTS_FILE = './models/words_segmenter_weights.hdf5'
ARCHITECTURE_FILE = './models/words_segmenter_architecture.json'
NAME = 'Characters_segmenter'


def modelExists():
    weightsExists = os.path.exists(WEIGHTS_FILE)
    architectureExists = os.path.exists(ARCHITECTURE_FILE)
    return weightsExists and architectureExists
