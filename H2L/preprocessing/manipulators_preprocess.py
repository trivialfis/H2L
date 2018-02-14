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

from skimage import io, transform
from random import shuffle
from tqdm import tqdm
import numpy as np
import cv2
import os
import pickle
from . import reform

from ..evaluator import h2l_debug

debugger = h2l_debug.h2l_debugger()

IMAGES_PATH = '../resource/characters/'
TARGET_FILE = '../resource/characters_pkl/characters'
VALIDATION_FILE = '../resource/characters_pkl/validation.pkl'
TASK_NUM = 8
HEIGHT = 48
VALIDATION_SIZE = 500


def mapping(path):
    manipulators = [d for d in os.listdir(path)
                    if os.path.isdir(path+d)]
    manipulators.sort()
    category = {}
    categoryPrint = {}
    for i in range(len(manipulators)):
        categoryPrint[i] = manipulators[i]
        category[manipulators[i]] = i
    f = open('../resource/character_map', 'w')
    f.write(str(categoryPrint))
    f.close()
    return category


def loadImagesInfo(path):
    print('Loading info')
    imagesPath = [os.path.join(root, img)
                  for root, subdirs, filenames in os.walk(path)
                  for img in filenames]
    category = mapping(path)
    # print(imagesPath[0])
    imagesInfo = [(imgPath, category[imgPath.split('/')[-2]])
                  for imgPath in imagesPath]
    print('Info loaded')
    return imagesInfo


def loadImages(imagesInfo, bar):
    labeledImages = []
    for path, label in imagesInfo:
        image = np.array(io.imread(path), dtype=np.uint8)
        labeledImages.append((image, label))
        bar.update(1)
    return labeledImages


def resize(labeledImages, bar):
    outputShape = (HEIGHT, HEIGHT)
    result = []
    for image, label in labeledImages:
        bar.update(1)
        try:
            resizedImage = transform.resize(image, outputShape)
            # resizedImage = np.array(binarize(resizedImage), dtype=np.float32)

            # resizedImage[0, :] = 0
            # resizedImage[-1, :] = 0
            # resizedImage[:, 0] = 0
            # resizedImage[:, -1] = 0
            resizedImage = reform.removeEdge(resizedImage)
            debugger.image_info(prefix='resize:',
                                image=resizedImage)
            resizedImage = [
                cv2.threshold(
                    img, 0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )[1]
                for img in resizedImage
            ]
            resizedImage = resizedImage.reshape(HEIGHT, HEIGHT, 1)
        except ValueError:
            print('Resize error, image shape: ', image.shape)
            continue
        result.append((resizedImage, label))
    return result


def save(labeledImages, targetFile):
    with open(targetFile, 'wb') as f:
        pickle.dump(labeledImages, f)


def start():
    imagesInfo = loadImagesInfo(IMAGES_PATH)
    shuffle(imagesInfo)
    bar = tqdm(total=2*len(imagesInfo), unit=' steps')
    validationInfo = imagesInfo[-VALIDATION_SIZE:]
    imagesInfo = imagesInfo[:-VALIDATION_SIZE]
    length = len(imagesInfo)
    tasks = [imagesInfo[(length)//TASK_NUM*i: length//TASK_NUM*(i+1)]
             for i in range(TASK_NUM)]
    for i in range(TASK_NUM):
        labeledImages = loadImages(tasks[i], bar)
        labeledImages = resize(labeledImages, bar)
        # images, labels = zip(*labeledImages)
        save(labeledImages, TARGET_FILE+str(i)+'.pkl')
    print('Generating validation data')
    bar = tqdm(total=len(validationInfo), unit=' steps')
    labeledImages = loadImages(validationInfo, bar)
    labeledImages = resize(labeledImages, bar)
    # images, labels = zip(*labeledImages)
    save(labeledImages, VALIDATION_FILE)
