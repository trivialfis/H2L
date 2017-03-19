'''
File:          slantCorrect.py
Author:        fis
Created:       Feb 13 2017
Last modified: Feb 20 2017

Description:
Cite:
A new normalization technique for cursive handwritten words
Alessandro Vinciarelli * , Juergen Luettin 1
IDIAP ± Institut Dalle Molle d'Intelligence, Arti®cielle Perceptive,
Rue du Simplon 4, CP592, 1920 Martigny, Switzerland
Received 19 October 2000; received in revised form 26 February 2001
'''
from skimage import filters
from skimage.transform import AffineTransform, warp
import numpy as np


def correctSlant(image, tolerance=0.15, angleRange=np.pi/4, step=0.1,
                 label=None):
    '''
    Words slant correction
    Parameter:
        image: one single word image
        tolerance: for near continuous stroke
        angleRange: the range of angle to try
        step:
        label: [(x, y)] segmentation points
    return: a slant corrected image, width is doubled
    '''
    def binarize(image):
        value = filters.threshold_isodata(image)
        mask = image < value
        mask = np.array(mask, dtype=np.float32)
        return mask

    def binarizeRevert(image):
        value = filters.threshold_otsu(image)
        mask = image > value
        mask = np.array(mask, dtype=np.float32)
        return mask

    def padding(image, label=None):
        '''Expand the image width so that shearing won't cut the image'''
        rows, columns = image.shape
        paddedImage = np.zeros((rows, 2*columns))
        colStart = columns//2
        paddedImage[:, colStart:colStart+columns] = image
        if label is not None:
            label = [(point[0]+colStart, point[1]) for point in label]
            return paddedImage, label
        else:
            return paddedImage

    def getVariance(image):
        rows, columns = image.shape
        continuous = []
        for col in range(columns):
            summation = np.sum(image[:, col])
            top = 0
            bottom = -1
            for r in range(rows):
                if image[r, col] != 0:
                    top = r
                    break
            for r in range(rows):
                if image[-r, col] != 0:
                    bottom = rows - r
                    break
            if bottom - top + 1 == 0:
                continue
            if 1.0 - (summation / (bottom - top + 1)) < tolerance:
                continuous.append(col)
        var = 0
        for col in continuous:
            temp = np.sum(image[:, col])**2
            var += temp
        return var

    if label is not None:
        image, label = padding(image, label)
        # print(label)
        mask = np.zeros(image.shape)
        for point in label:
            # point[0]: x, point[1]: y
            # print(type(point[1]), type(point[0]))
            mask[point[1], point[0]] = 1
    else:
        image = padding(image)

    angles = np.arange(-angleRange, angleRange, step)
    maxVariance = getVariance(image)
    corrected = image

    slantMatrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    for angle in angles:
        matrix = np.array([[1, np.tan(angle), 0],
                           [0, 1,             0],
                           [0, 0,             1]])
        sheared = warp(image,
                       AffineTransform(matrix=matrix),
                       preserve_range=True)
        # sheared = binarizeRevert(sheared)
        var = getVariance(sheared)
        if var > maxVariance:
            maxVariance = var
            corrected = sheared
            slantMatrix = matrix
    if label is not None:
        mask = warp(mask, AffineTransform(matrix=slantMatrix),
                    preserve_range=True)
        segmentations = []
        for i in range(mask.shape[1]):
            if np.sum(mask[:, i]) > 0 and i-1 not in segmentations:
                segmentations.append(i)
        return (corrected, segmentations)
    else:
        return corrected
