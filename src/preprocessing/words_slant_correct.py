'''
File:          word_slant_correct.py
Author:        fis
Created date:  Feb 22 2017
Last modified: Feb 22 2017
'''
from skimage import io
from preprocessing.reform import binarize
from normalization import slantCorrect

IMAGE_PATH = '../resource/words/'
XML_PATH = '../resource/xml/'
# TARGET_FILE = '../resource/base_words.pkl'
HEIGHT = 64

FRAGMENT_LENGTH = 5
SPACE = FRAGMENT_LENGTH // 2
RAND_RANGE = 50


def start(wordList):
    for wordID, segmentationPoints in wordList:
        pathElements = wordID.split('-')
        localPath = (pathElements[0] + '/'
                     + pathElements[0] + '-' + pathElements[1] + '/'
                     + wordID + '.png')
        fullPath = IMAGE_PATH + localPath
        image = io.imread(fullPath)
        image = binarize(image, mode='less', threshold='isodata')
        image = slantCorrect.correctSlant(image)
        io.imsave(arr=image, fname=fullPath)
