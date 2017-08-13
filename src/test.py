#!/usr/bin/python3
from skimage import io
import cv2
import numpy as np
from preprocessing import reform
from normalization import slantCorrect

from matplotlib import pyplot as plt
from evaluator import h2l_debug

import sys

np.set_printoptions(threshold=np.nan)
h2l_debug.H2L_DEBUG = True
debugger = h2l_debug.h2l_debugger()


def testHeuristicSegmenter():
    from evaluator import heuristicSegmenter
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
    from evaluate import heursiticGenerate
    imageFile = '../resource/test/testform.jpg'
    image = io.imread(imageFile)
    result = heursiticGenerate(image)
    print(result)


def testRecognizer():
    from evaluator import characterRecognizer
    imageFile = '../resource/test/character.png'
    image = io.imread(imageFile)
    image = image.reshape((1, ) + image.shape + (1, ))
    rc = characterRecognizer.recognizer()
    result = rc.predict(image)
    print(result)


def testLineSegmenter():
    from evaluator.LineSegment import LineSegment
    imageFile = '../resource/test/line2.png'
    image = cv2.imread(imageFile, 0)
    image = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    result = LineSegment.segment(image)
    debugger.display('Number of lines:', len(result))
    count = 0
    for img in result:
        cv2.imwrite(img=img, filename='line_segment' + str(count) + '.png')
        count += 1


def testExtractDoc():
    from evaluator import extractDocument
    imageFile = '../resource/test/testform.jpg'
    image = io.imread(imageFile)
    result = extractDocument.drawContours(image)
    plt.imshow(result, cmap='gray')
    plt.show()


def testSegmentRecognize():
    from evaluator import characterRecognizer
    from evaluator import heuristicSegmenter
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
    from evaluate import heursiticGenerate
    imageFile = '../resource/test/word_test_image.png'
    image = io.imread(imageFile)
    image = reform.binarize(image, mode='greater')
    heursiticGenerate(image)


if __name__ == '__main__':
    args_map = {'ls': testLineSegmenter,
                'he': testHeuristicEvaluate}
    try:
        try:
            action = sys.argv[1]
        except IndexError:
            print('Usage: ./test.py <action>')
        try:
            args_map[sys.argv[1]]()
        except KeyError:
            print('Available tests:\n',
                  'ls: testLineSegmenter\n',
                  'he: testHeuristicEvaluate\n')
        # testLineEvaluate()
        # testSegmentRecognize()
        # testHeuristicEvaluate()
        # testRecognizer()
        # testHeuristicSegmenter()
        # testLineSegmenter()
        # testExtractDoc()
    except KeyboardInterrupt:
        print('\nExit')
