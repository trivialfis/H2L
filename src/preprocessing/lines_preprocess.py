#!/usr/bin/env python3
#
# Copyright © 2017, 2018 Fis Trivial <ybbs.daans@hotmail.com>
#
# This file is part of H2L.
#
# H2L is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# H2L is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with H2L.  If not, see <http://www.gnu.org/licenses/>.
#

from xml.etree import ElementTree as ET
from skimage import io, filters  # , transform
from PIL import Image
import numpy as np
from random import shuffle, randint
import os
import pickle
from tqdm import tqdm


IMAGE_PATH = '../resource/lines/'
XML_PATH = '../resource/xml/'
TARGET_PATH = '../resource/lines_pkl/'
HEIGHT = 64

FRAGMENT_LENGTH = 5
SPACE = FRAGMENT_LENGTH // 2
RAND_RANGE = 50


def parseXML(xmlFilePath):
    '''
    Load and parse a single xml file
    Parameters:
        xmlFilePath: the path of the xml file
    return: A list containing tuples (image, points)

    Description:
    Only the white space between two words would be considered to be a
    segmentation point for the reason that the edges of line images are
    too close to the edges of words, which simply means the start and the
    end points of lines won't be included in the result tuple list.
    '''
    form = ET.parse(xmlFilePath).getroot()
    handwritten = form.getchildren()[1]
    lines = handwritten.getchildren()
    lines = [line for line in lines if line.attrib['segmentation'] == 'ok']
    matched = []
    for line in lines:
        wordsList = line.getchildren()
        segmentationPoints = []
        wordsBoundaries = []
        wordsLeftBoundary = []  # the x value of characters for every words
        for word in wordsList:
            if word.tag == 'word':
                characters = word.getchildren()
                characterX = [int(c.attrib['x']) for c in characters]
                characterWidth = [int(c.attrib['width']) for c in characters]
                try:
                    left = min(characterX)
                # Some words in xmls are empty, but the images are fine
                except ValueError:
                    print('Empty word found\t| form id: ', form.attrib['id'],
                          '\t| line id:', line.attrib['id'],
                          '\t| word id: ', word.attrib['id'],
                          '\t|')
                    continue
                else:
                    wordsLeftBoundary.append(left)
                    rightX = max(characterX)
                    rightXIndex = characterX.index(rightX)
                    rightXWidth = characterWidth[rightXIndex]
                    right = rightX + rightXWidth
                    wordsBoundaries.append((left, right))
            else:
                continue
        lastTail = 0
        lineLeftBoundary = min(wordsLeftBoundary)
        for (a, b) in wordsBoundaries:
            middle = (lastTail + a-lineLeftBoundary) / 2
            lastTail = b-lineLeftBoundary
            segmentationPoints.append(middle)
        # ditch the left most point
        matched.append((line.attrib['id'], segmentationPoints[1:]))
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
    print(len(matched), ' lines in total ')
    return matched


def extract(lines):
    '''
    Extract fragments from lines

    Description:
    The data flow is as follow: the lines send in as a list containing multiple
    tuples like [(image, label)], then we extract all the fragments out of the
    images and label it with 1 or 0 as in tuple(fragment, 0). At last unzip it
    into two list then combine two list into on tuple, which means at the end
    the return value would has the form ([fragments], [labels])
    '''
    fragments = []
    print('Extracting fragments')
    bar = tqdm(total=len(lines), unit=' images')
    for line in lines:
        foundPointsCount = 0
        image, label = line[0], line[1]
        for k in range(0, image.shape[1]):
            if k < SPACE:
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
        bar.update(1)
    filteredFragments = []
    for f in fragments:
        if f[1]:
            filteredFragments.append(f)
        elif f[1] == 0 and randint(0, RAND_RANGE) == 0:
            filteredFragments.append(f)
        else:
            continue
    images, labels = zip(*filteredFragments)
    images, labels = np.array(images), np.array(labels)
    return (images, labels)


def resize(labeledImage, uniformedHeight=28):
    '''
    Resize a single image using PIL module. In the procedure, the segmentation
    points will also be resize accordingly

    Parameters:
        labeledImage:    A single image labeled with segmentation points, it's
                         format looks like (image, points)
        uniformedHeight: The images and points are resized based on the
                         required height, which is set default to 28.

    return:          The resized image, label tuple, like the parameter.
    '''
    image, label = labeledImage
    width, height = image.shape
    ratio = uniformedHeight / width
    # outputShape = (uniformedHeight, round(ratio * height))
    outputShape = (round(ratio*height), uniformedHeight)  # width, height
    # image = transform.resize(image, outputShape)
    image = np.array(
        Image.fromarray(image).resize(
            outputShape,
            Image.ANTIALIAS)
    )
    label = [round(l * ratio) for l in label]
    return (image, label)


def binarize(image):
    '''Binarize a single image using skimage library'''
    value = filters.threshold_isodata(image)
    mask = image < value
    mask = np.array(mask, dtype=np.float32)
    return mask


def save(labeledImagesList, targetFile):
    '''Save the matched images into a pkl file'''
    with open(targetFile, 'ab') as target:
        pickle.dump(labeledImagesList, target)
        print('Dumped', targetFile)


def start():
    linesList = loadXMLs(XML_PATH)
    imagesBuffer = []
    if not os.path.exists(TARGET_PATH):
        os.mkdir(TARGET_PATH)
    print('Processing labeled images')
    bar = tqdm(total=len(linesList), unit=' images')
    for lineID, segmentationPoints in linesList:
        pathElements = lineID.split('-')
        imagePath = (pathElements[0] + '/'
                     + pathElements[0] + '-' + pathElements[1] + '/'
                     + lineID + '.png')
        fullPath = IMAGE_PATH + imagePath
        image = io.imread(fullPath)
        image = binarize(image)
        labeledImage = (image, segmentationPoints)
        labeledImage = resize(labeledImage, HEIGHT)
        imagesBuffer.append(labeledImage)
        bar.update(1)
    shuffle(imagesBuffer)
    targetFile = TARGET_PATH + 'lines.pkl'
    # imagesBuffer = extract(imagesBuffer)
    batch = len(imagesBuffer) // 8
    tasts = [imagesBuffer[batch*i: batch*(i+1)]
             for i in range(7)]
    tasts.append(imagesBuffer[batch*7:])
    for target in tasts:
        fragments = extract(target)
        save(fragments, targetFile)
    print('Done')


if __name__ == '__main__':
    start()
