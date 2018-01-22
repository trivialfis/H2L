#!/usr/bin/env python3
'''
File:          heuristic_segmenter.py
Author:        fis
Created:       Feb 22 2017
Last modified: Aug 15 2017
'''
import numpy as np
from normalization import image_utils
from evaluator import line_segmenter
from evaluator import h2l_debug
import cv2


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
