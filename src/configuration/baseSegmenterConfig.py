'''
File: segmenter_trainer.py
Author:        fis
Created:       25 Jan 2017
Last modified:  2 Feb 2017

Description:
The configuration module for segmeters
'''
BASE_ARCHITECTURE_FILE = './models/base_segmenter_architecture.json'
BASE_WEIGHTS_FILE = './models/base_segmenter_weights.hdf5'
BASE_NAME = 'Base_segmenter'

HEIGHT = 64
WIDTH = round(HEIGHT * 4/5)
FRAGMENT_LENGTH = 5
SPACE = FRAGMENT_LENGTH // 2
AW = round(HEIGHT)  # acceptable width

BATCH_SIZE = 128
EPOCH = 150
IMG_ROWS, IMG_COLS = HEIGHT, FRAGMENT_LENGTH
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

ARCHITECTURE_FILE = './models/words_segmenter_architecture.json'
WEIGHTS_FILE = './models/words_segmenter_weights.hdf5'
NAME = 'Characters_segmenter'


def baseModelExists():
    import os
    archtectureExists = os.path.exists(BASE_ARCHITECTURE_FILE)
    weightsExists = os.path.exists(BASE_WEIGHTS_FILE)
    return archtectureExists and weightsExists


def wordsModelExists():
    import os
    archtectureExists = os.path.exists(ARCHITECTURE_FILE)
    weightsExists = os.path.exists(WEIGHTS_FILE)
    return archtectureExists and weightsExists
