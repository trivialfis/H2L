'''
File:          evaluate.py
Author:        fis
Created:       Feb 17 2017
Last modified: Mar 16 2017
'''
from evaluator import baseCharSegmenter, characterRecognizer, wordSegmenter, heuristicSegmenter
from configuration import characterRecognizerConfig as crconfig
from preprocessing import reform
import numpy as np

from preprocessing.reform import saveImages

ws = wordSegmenter.segmenter()
bcs = baseCharSegmenter.segmenter()
hs = heuristicSegmenter.segmenter()
cr = characterRecognizer.recognizer()

from matplotlib import pyplot as plt


def generate(image, preprocess=False):
    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))
    image = reform.rescale(image, 64)
    wordImages = ws.segment(image)
    charactersList = []
    saveImages.counter = 0
    print('Word num ', len(wordImages))
    for word in wordImages:
        # characterImages = bcs.segment(word)  # [2:-2]
        characterImages = hs.segment(word)
        if len(characterImages) == 0:
            continue
        characterImages = [reform.resize(
            char, (crconfig.IMG_COLS, crconfig.IMG_COLS))
                           for char in characterImages]
        saveImages(characterImages, prefix='bef')
        characterImages = [reform.removeEdge(char)
                           for char in characterImages]
        saveImages(characterImages, prefix='aft')
        characterImages = [char.reshape(char.shape + (1, ))
                           for char in characterImages]
        characterImages = np.array(characterImages, dtype=np.float32)
        characters = cr.predict(characterImages)
        print('number of prediction', len(characters))
        characters.append(' ')
        charactersList += characters
        print('number of character images', len(charactersList))
        print(charactersList)
        characters = ''.join(charactersList)
    return characters


def heursiticGenerate(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))
    image = reform.rescale(image, 64)
    characterImages = hs.segment(image)
    print('char num: ', len(characterImages))
    characterImages = [reform.resize(
        char, (crconfig.IMG_COLS, crconfig.IMG_COLS)
    ) for char in characterImages
    ]
    saveImages.counter = 0
    saveImages(characterImages, prefix='bef')
    # characterImages = [reform.removeEdge(char)
    #                    for char in characterImages]
    # saveImages(characterImages, prefix='after')
    characterImages = [char.reshape(char.shape + (1, ))
                       for char in characterImages]
    characterImages = np.array(characterImages, dtype=np.float32)
    characters = cr.predict(characterImages)
    return characters


if __name__ == '__main__':
    generate()
