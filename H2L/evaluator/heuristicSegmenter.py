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

import numpy as np
import cv2

from ..normalization import image_utils
from . import line_segmenter
from . import h2l_debug


debugger = h2l_debug.h2l_debugger()


class segmenter(object):
    def __extractCharacters(self, image, segmentationPoints, HEIGHT):
        if len(image.shape) != 2:
            raise ValueError('Expected image shape (x, y), got ', image.shape)
        lastPoint = 0
        characterList = []
        for i in range(1, len(segmentationPoints)):
            character = np.zeros((HEIGHT, HEIGHT))
            rows, cols = image.shape
            length = segmentationPoints[i] - segmentationPoints[lastPoint]

            if length <= 0:
                raise ValueError(
                    'Negative length, sp[i]:'+str(
                        segmentationPoints[i]
                    )+' lp:'+str(
                        segmentationPoints[lastPoint]
                    )
                )
            if length > HEIGHT:
                ratio = HEIGHT / length
            else:
                ratio = 1
            resized = cv2.resize(
                image[:,
                      segmentationPoints[lastPoint]:
                      segmentationPoints[lastPoint]+length],
                dsize=(0, 0),
                fx=ratio, fy=ratio,
                interpolation=cv2.INTER_NEAREST
            )
            colStart = (HEIGHT - resized.shape[1]) // 2
            rowStart = (HEIGHT - resized.shape[0]) // 2
            character[
                rowStart:rowStart+resized.shape[0],
                colStart:colStart+resized.shape[1]
            ] = resized
            lastPoint = i
            characterList.append(character)
        return characterList

    def segment(self, image):
        if len(image.shape) != 2:
            raise ValueError('expected image shape (x, y), got ', image.shape)
        height, width = image.shape
        if width <= height*0.2:
            return image
        i = 0
        segmentationPoints = []
        while i < width:
            if np.sum(image[:, i]) < 1.0:
                j = i + 1
                # the j < width run before sum is important
                while j < width and np.sum(image[:, j]) < 2.0:
                    j += 1
                mid = (i + j) // 2
                i = j
                segmentationPoints.append(mid)
            i += 1

        characterList = self.__extractCharacters(image,
                                                 segmentationPoints,
                                                 HEIGHT=height)
        return characterList


def segment(image):
    temp = cv2.transpose(image.copy())
    temp = image_utils.remove_edges(temp)
    characters = line_segmenter.segment(temp)
    return characters
