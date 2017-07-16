'''
File:          evaluate.py
Author:        fis
Created:       Feb  17 2017
Last modified: July 17 2017
'''
from evaluator import heuristicSegmenter, wordSegmenter  # , baseCharSegmenter
from evaluator import characterRecognizer
from evaluator import toLaTeX
from evaluator import extractDocument
from evaluator.LineSegment import LineSegment
from configuration import characterRecognizerConfig as crconfig
from preprocessing import reform
# import xml.etree.ElementTree as ET
import numpy as np
# from skimage import filters, color
from preprocessing.reform import saveImages
from matplotlib import pyplot as plt

# import cv2

import warnings
warnings.filterwarnings('ignore')

# ws = wordSegmenter.segmenter()
# bcs = baseCharSegmenter.segmenter()
hs = heuristicSegmenter.segmenter()
cr = characterRecognizer.recognizer()


def generate(image, preprocess=False):
    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))
    image = reform.rescale(image, 64)
    wordImages = wordSegmenter.segment(image)
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
    "Generate LaTeX pdf from image using heuristic methods."

    def findMid(line):
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(line.shape[0]):
            if np.sum(line[i, :]) >= 1.0:
                top = i
                break
        for i in range(line.shape[0]):
            if np.sum(line[-i, :]) >= 1.0:
                bottom = line.shape[0] - i + 1
                break
        middle = (top + bottom) / 2
        return middle

    def isSuper(character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(1, character.shape[0]):
            if np.max(character[-i, :]) != 0:
                bottom = character.shape[0] - i
                break
        if bottom < middle:
            return True
        else:
            return False

    def isSub(character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y) ,got ' +
                             str(character.shape))
        for i in range(character.shape[0]):
            if np.max(character[i, :]) != 0:
                top = i
                print('Top: ', top)
                break
        if top > middle:
            return True
        else:
            return False

    def segmentCharacters(line):
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        characterImages = hs.segment(line)
        middle = findMid(line)
        superFlag = [isSuper(char, middle) for char in characterImages]  # exp
        subFlag = [isSub(char, middle) for char in characterImages]  # index
        characterImages = [reform.removeEdge(char)
                           for char in characterImages]
        characterImages = [reform.resize(
            char, (crconfig.IMG_ROWS, crconfig.IMG_COLS))
                           for char in characterImages]
        characterImages = [reform.binarize(char, mode='greater')
                           for char in characterImages]
        characterImages = [char.reshape(char.shape+(1, ))
                           for char in characterImages]
        characterImages = np.array(characterImages, dtype=np.float32)
        characters = cr.predict(characterImages)
        print('Evaluate::segmentCharacters::characters ', characters)
        print('Evaluate::segmentCharacters::length ', len(characters))
        equation = ''
        for i in range(len(characters)):
            if superFlag[i]:
                char = '^' + characters[i] + ' '
                equation += char
                print('^', characters[i], end='')
            elif subFlag[i]:
                char = '_' + characters[i] + ' '
                equation += char
                print('_', characters[i], end='')
            else:
                char = characters[i] + ' '
                equation += char
                print(characters[i], end='')
        print('\nEvaluate::segmentCharacters end\n\n')
        return equation

    if len(image.shape) != 3:
        raise ValueError('Expected image with shape (x, y, z), got ' +
                         str(image.shape))
    image = extractDocument.drawContours(image)  # Extract documentation image
    # image = filters.gaussian_filter(image, 1)
    print('after extractDocument', image.shape)
    plt.imshow(image, cmap='gray')
    plt.show()

    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))
    lineImages = LineSegment.segment(image)
    lineImages = [reform.rescale(line, 64) for line in lineImages]
    print('Evaluate::After line segmentation')
    for line in lineImages:
        plt.imshow(line, cmap='gray')
        plt.show()
    equations = []
    for line in lineImages:
        equations.append(segmentCharacters(line))
    toLaTeX.transoform(equations)


if __name__ == '__main__':
    heursiticGenerate()
    # generate()
