#!/bin/python3
'''
File:          words_preprocess.py
Author:        fis
Created date:  Jan 30 2017
Last modified: Feb 21 2017

Description:
Preprocess the words images for network training. The procedures include
binarizing, resizing the line images and matching it with the corresponding
segmentation points obtained from xml files.
'''
from xml.etree import ElementTree as ET
from skimage import io, transform
import numpy as np
from random import shuffle, randint
import os
import pickle
from normalization import slantCorrect
from preprocessing.reform import binarize


IMAGE_PATH = '../resource/words/'
XML_PATH = '../resource/xml/'
TARGET_FILE = '../resource/base_words.pkl'
HEIGHT = 64

FRAGMENT_LENGTH = 5
SPACE = FRAGMENT_LENGTH // 2
RAND_RANGE = 50


def parseXML(xmlFilePath):
    '''
    Get the segmentation points from a xml file
    Parameters:
        xmlFilePath: the path of a xml file
    return: [(id, segmentation points)], containing multiple tuples
    '''
    wordList = []    # All words in the xml file
    form = ET.parse(xmlFilePath).getroot()
    handwritten = form.getchildren()[1]
    lines = handwritten.getchildren()
    for line in lines:
        words = line.getchildren()
        for word in words:
            wordList.append(word)

    matched = []
    for word in wordList:
        charBoundaries = []
        charLeft = []
        charRight = []
        charTop = []
        segmentationPoints = []
        characters = word.getchildren()
        if word.tag == 'word' and len(characters) == len(word.attrib['text']):
            for c in characters:
                charLeft.append(int(c.attrib['x']))
                charRight.append(int(c.attrib['x']) + int(c.attrib['width']))
                charTop.append(int(c.attrib['y']))
            minTop = min(charTop)
            charTop = [(top - minTop) for top in charTop]

            leftBound = min(charLeft)
            charBoundaries = zip(charLeft, charRight)
            lastEnd = 0
            for ((a, b), t) in zip(charBoundaries, charTop):
                middle = (lastEnd + a - leftBound) // 2
                lastEnd = b - leftBound
                segmentationPoints.append((middle, t))  # (x, y)
            matched.append((word.attrib['id'], segmentationPoints))
        else:
            continue
    return matched


def loadXMLs(xmlPath):
    xmlFiles = [xml for root, subdirs, files
                in os.walk(xmlPath) for xml in files]
    print(len(xmlFiles), ' xml files in total')
    xmlFilesPath = []
    for xml in xmlFiles:
        path = XML_PATH + xml.split('-')[0] + '/' + xml
        xmlFilesPath.append(path)
    matched = []
    for xmlFile in xmlFilesPath:
        parseResult = parseXML(xmlFile)
        matched += parseResult
    print(len(matched), ' words in total ')
    return matched


def extract(words):
    '''
    Extract fragments from words
    Parameters:
        words: [(image, label)], containing multiple tuples

    return: [(fragment, label)], containing multiple tuples
    '''
    fragments = []
    for word in words:
        foundPointsCount = 0
        image, label = word[0], word[1]
        for k in range(0, image.shape[1]):
            if image.shape[1] < FRAGMENT_LENGTH:
                segment = np.zeros((HEIGHT, FRAGMENT_LENGTH), dtype=np.float32)
                segment[:, 0:image.shape[1]] = image
            elif k < SPACE:
                segment = np.zeros((HEIGHT, FRAGMENT_LENGTH), dtype=np.float32)
                segment[:, SPACE-k:] = image[:, 0:SPACE+k+1]
            elif image.shape[1]-k-1 < SPACE:
                segment = np.zeros((HEIGHT, FRAGMENT_LENGTH), dtype=np.float32)
                temp = FRAGMENT_LENGTH - SPACE + image.shape[1] - k - 1
                segment[:, 0:temp] = image[:, k-SPACE:]
            else:
                segment = np.array(image[:, k-SPACE:k+SPACE+1],
                                   dtype=np.float32)
            fragments.append(
                (segment.reshape(HEIGHT, FRAGMENT_LENGTH, 1),
                 1 if k in label else 0))
            if foundPointsCount == len(label):
                break
            else:
                continue
    filteredFragments = []
    count = 0
    for f in fragments:
        if f[1]:
            filteredFragments.append(f)
            count += 1
        elif f[1] == 0 and randint(0, RAND_RANGE) == 0:
            filteredFragments.append(f)
        else:
            continue
    return filteredFragments


def resize(labeledImage, uniformedHeight=64):
    '''
    Resize a single image using PIL module. In the procedure, the segmentation
    points will also be resize accordingly

    Parameters:
        labeledImage:    A single image labeled with segmentation points, it's
                         format looks like (image, points)
        uniformedHeight: The images and points are resized based on the
                         required height, which is set default to 28.

    return: The resized image, label tuple, like the parameter.
    '''
    image, label = labeledImage
    # width, height = image.shape
    rows, cols = image.shape
    ratio = uniformedHeight / rows
    outputShape = (uniformedHeight, round(ratio * cols))
    # outputShape = (round(ratio*height), uniformedHeight)  # width, height
    image = transform.resize(image, outputShape)
    # image = np.array(
    #     Image.fromarray(image).resize(
    #         outputShape,
    #         Image.ANTIALIAS)
    # )
    # print(type(label))
    label = [round(l*ratio) for l in label]
    return (image, label)


def correctSlant(labeledImage):
    '''The function is not used'''
    image, label = labeledImage
    slantedImage, segmentationPoints = slantCorrect.correctSlant(image,
                                                                 label=label)
    # print('Segmentation points: ', segmentationPoints)
    return (slantedImage, segmentationPoints)


def save(labeledImagesList, targetFile=TARGET_FILE):
    '''Save the matched images into a pkl file'''
    with open(targetFile, 'wb') as target:
        pickle.dump(labeledImagesList, target)
        print('Dumped', targetFile)


def start(wordsList):
    imagesBuffer = []
    for wordID, segmentationPoints in wordsList:
        pathElements = wordID.split('-')
        imagePath = (pathElements[0] + '/'
                     + pathElements[0] + '-' + pathElements[1] + '/'
                     + wordID + '.png')
        fullPath = IMAGE_PATH + imagePath
        image = io.imread(fullPath)
        image = binarize(image, mode='less', threshold='isodata')
        labeledImage = (image, segmentationPoints)
        labeledImage = correctSlant(labeledImage)
        labeledImage = resize(labeledImage, HEIGHT)
        imagesBuffer.append(labeledImage)
    shuffle(imagesBuffer)
    # save(imagesBuffer, TARGET_FILE[:-4] + 't.pkl')
    imagesBuffer = extract(imagesBuffer)
    return imagesBuffer
