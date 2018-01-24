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
