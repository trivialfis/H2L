'''
File: lines_segmenter_trainer.py
Author:        fis
Created:       Jan 27 2017
Last modified: Feb 24 2017

Description:
The configuration module for lines segmeters
'''
ARCHITECTURE_FILE = './models/word_segmenter_architecture.json'
WEIGHTS_FILE = './models/word_segmenter_weights.hdf5'
NAME = 'Words_segmenter'

FRAGMENT_LENGTH = 5
SPACE = FRAGMENT_LENGTH // 2
HEIGHT = 64

BATCH_SIZE = 256
EPOCH = 120

IMAGE_ROWS, IMAGE_COLS = HEIGHT, FRAGMENT_LENGTH
INPUT_SHAPE = (IMAGE_ROWS, IMAGE_COLS, 1)


def modelExists():
    import os
    archtectureExists = os.path.exists(ARCHITECTURE_FILE)
    weightsExists = os.path.exists(WEIGHTS_FILE)
    return archtectureExists and weightsExists
