#!/bin/python3
from skimage import io, filters
from skimage.transform import rescale  # , resize
import numpy as np
from configuration import baseSegmenterConfig as config
from evaluator import baseCharSegmenter, characterRecognizer, wordSegmenter
from evaluator import heuristicSegmenter
from evaluator import extractDocument
from evaluator.LineSegment import LineSegment
from preprocessing import reform
from normalization import slantCorrect
from evaluate import generate, heursiticGenerate

from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)


class processor(object):
    def binarize(self, image):
        value = filters.threshold_isodata(image)
        mask = image < value
        mask = np.array(mask, dtype=np.float32)
        return mask

    def resize(self, image):
        uniformedHeight = config.HEIGHT
        rows, columns = image.shape
        ratio = uniformedHeight / rows
        image = rescale(image, ratio)
        return image

    def prep(self, image):
        image = self.resize(image)
        print(image.shape)
        return image


def testCharSegmenter():
    imageFile = '../resource/word_test_image.png'
    image = io.imread(imageFile)
    image = reform.binarize(image, mode='greater')
    p = processor()
    image = p.prep(image)
    kevin = baseCharSegmenter.segmenter()
    points = kevin.segment(image)
    image = slantCorrect.correctSlant(image)
    for p in points:
        image[:, p] = 1.0
    io.imsave(arr=image, fname='result.png')
    print(points)


def testWordSegmenter():
    imageFile = '../resource/l.png'
    image = io.imread(imageFile)
    image = reform.rescale(image, 28)
    image = reform.binarize(image, mode='less', threshold='isodata')
    segmenter = wordSegmenter.segmenter()
    words = segmenter.segment(image)
    for i in range(len(words)):
        io.imsave(arr=words[i], fname='l'+str(i)+'.png')


def testEvaluate():
    imageFile = '../resource/word_test_image.png'
    image = io.imread(imageFile)
    image = reform.binarize(image, mode='less', threshold='isodata')
    result = generate(image)
    print(result)


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
        testLineEvaluate()
        # testSegmentRecognize()
        # testHeuristicEvaluate()
        # testRecognizer()
        # testCharSegmenter()
        # testEvaluate()
        # testHeuristicSegmenter()
        # testLineSegmenter()
        # testExtractDoc()
    except KeyboardInterrupt:
        print('\nExit')
