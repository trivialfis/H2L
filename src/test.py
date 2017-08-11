#!/usr/bin/python3
from skimage import io
import numpy as np
from evaluator import characterRecognizer
from evaluator import heuristicSegmenter
from evaluator import extractDocument
from evaluator.LineSegment import LineSegment
from preprocessing import reform
from normalization import slantCorrect
from evaluate import heursiticGenerate

from matplotlib import pyplot as plt
from evaluator import h2l_debug

np.set_printoptions(threshold=np.nan)
h2l_debug.H2L_DEBUG = True


def testHeuristicSegmenter():
    imageFile = '../resource/word_test_image.png'
    image = io.imread(imageFile)
    image = reform.binarize(image, mode='less', threshold='isodata')
    image = slantCorrect.correctSlant(image)
    image = reform.rescale(image, 64)
    segmentation = heuristicSegmenter.overSegment(image)
    print('points: ', segmentation)
    for p in segmentation:
        image[:, p] = 1
    io.imsave(arr=image, fname='result.png')


def testHeuristicEvaluate():
    imageFile = '../resource/test/testform.jpg'
    image = io.imread(imageFile)
    result = heursiticGenerate(image)
    print(result)


def testRecognizer():
    imageFile = '../resource/test/character.png'
    image = io.imread(imageFile)
    image = image.reshape((1, ) + image.shape + (1, ))
    rc = characterRecognizer.recognizer()
    result = rc.predict(image)
    print(result)


def testLineSegmenter():
    imageFile = '../resource/test/character.png'
    image = io.imread(imageFile)
    image = reform.binarize(image, mode='greater', threshold='isodata')
    result = LineSegment.LineSegment(image)
    print(len(result))
    count = 0
    for img in result:
        io.imsave(arr=img, fname='ls' + str(count) + '.png')


def testExtractDoc():
    imageFile = '../resource/test/testform.jpg'
    image = io.imread(imageFile)
    result = extractDocument.drawContours(image)
    plt.imshow(result, cmap='gray')
    plt.show()


def testSegmentRecognize():
    imageFile = '../resource/test/word_test_image.png'
    image = io.imread(imageFile)
    image = reform.binarize(image, mode='greater')
    hs = heuristicSegmenter.segmenter()
    characterImages = hs.segment(image)
    characterImages = [reform.removeEdge(img) for img in characterImages]
    cr = characterRecognizer.recognizer()
    characterImages = [reform.resize(char, outputShape=(48, 48))
                       for char in characterImages]
    characterImages = [char.reshape(char.shape + (1, ))
                       for char in characterImages]
    characterImages = np.array(characterImages, dtype=np.float32)
    characters = cr.predict(characterImages)
    print(characters)


def testLineEvaluate():
    imageFile = '../resource/test/word_test_image.png'
    image = io.imread(imageFile)
    image = reform.binarize(image, mode='greater')
    heursiticGenerate(image)


if __name__ == '__main__':
    try:
        # testLineEvaluate()
        # testSegmentRecognize()
        # testHeuristicEvaluate()
        testRecognizer()
        # testHeuristicSegmenter()
        # testLineSegmenter()
        # testExtractDoc()
    except KeyboardInterrupt:
        print('\nExit')
