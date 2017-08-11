'''
File:          transform.py
Author:        fis
Created:       Feb 14 2017
Last modified: Mar 14 2017
'''
from skimage import transform, filters
import numpy as np
from skimage import io

MODE = 'less'
THRESHOLD = 'isodata'
ROTATE_RANGE = np.pi / 12
SHEAR_RANGE = np.pi / 12
ZOOM_RANGE = 0.2


def removeEdge(image):
    '''Remove the white edges of the characters'''
    def detectRow(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i if reverse else i
            filled = np.sum(image[index, :])
            if filled <= 1.0:
                count += 1
            else:
                break
        return count

    def detectCol(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i if reverse else i
            filled = np.sum(image[:, index])
            if filled <= 1.0:
                count += 1
            else:
                break
        return count

    def subprocess(image):
        height, width = image.shape
        top = detectRow(image, height)
        left = detectCol(image, width)
        down = height - detectRow(image, height, True)
        right = width - detectCol(image, width, True)
        rows = down - top
        cols = right - left
        if rows < height * 0.1 or cols < width * 0.1:
            return image

        result = np.array(image[top: down+1, left: right+1],
                          dtype=np.uint8)
        return result

    if len(image.shape) != 2:
        raise ValueError('Image shape should be (x, y), found' +
                         str(image.shape))
    height, width = image.shape
    image = subprocess(image)
    length = max(image.shape)
    if (height, width) != image.shape:
        image = resize(image, outputShape=(length, length))
        image = rescale(image, height)
    return image


def saveImages(images, prefix=''):
    for im in images:
        saveImages.counter += 1
        io.imsave(arr=im, fname=prefix+str(saveImages.counter)+'.png')


def binarize(image, mode=MODE, threshold=THRESHOLD):
    thMapping = {
        'average': lambda image: np.average(image),
        'adaptive': lambda image: filters.threshold_local(image,
                                                          block_size=7),
        'isodata': lambda image: filters.threshold_isodata(image),
        'otsu': lambda image: filters.threshold_otsu(image),
        'min': lambda image: np.min(image) * 1.7
    }
    modeMapping = {
        'less': lambda image, value: image < value,
        'greater': lambda image, value: image > value
    }

    value = thMapping[threshold](image)
    mask = modeMapping[mode](image, value)
    mask = np.array(mask, dtype=np.uint8)

    return mask


def rescale(image, height, label=None):
    rows, cols = image.shape
    ratio = height / rows
    image = transform.rescale(image, ratio)
    if label is not None:
        label = [round(l*ratio) for l in label]
        return (image, label)
    else:
        return image


def resize(image, outputShape=(48, 48)):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ' +
                         str(image.shape))
    maxInput = max(image.shape)
    maxOutput = max(outputShape)
    resized = np.zeros(outputShape)
    if maxInput > maxOutput:
        image = transform.rescale(image, maxOutput/maxInput)
    rows, cols = image.shape
    rowStart = (outputShape[0] - rows) // 2
    colStart = (outputShape[1] - cols) // 2
    resized[rowStart:rowStart+rows, colStart:colStart+cols] = image
    return resized


def doubleColumns(image):
    '''
    Pad an image
    Parameters:
        image: An image array
    return : padded image
    '''
    rows, columns = image.shape
    paddedImage = np.zeros((rows, columns*2))
    colStart = columns//2
    paddedImage[:, colStart:colStart+columns] = image
    return paddedImage


def randomRotate(image, angleRange=ROTATE_RANGE, outputNum=1):
    rotated = []
    for i in range(outputNum):
        angle = np.random.uniform(-angleRange, angleRange)
        degree = angle * 180 / np.pi
        result = transform.rotate(image, degree, mode='constant', cval=0,
                                  resize=False)
        rotated.append(result)
    if outputNum == 1:
        rotated = rotated[0]
    return rotated


def randomShear(image, angleRange=SHEAR_RANGE, outputNum=1):
    sheared = []
    for i in range(outputNum):
        angle = np.random.uniform(-angleRange, angleRange)
        result = transform.warp(image, transform.AffineTransform(shear=angle))
        sheared.append(result)
    if outputNum == 1:
        sheared = sheared[0]
    return sheared


def randomZoom(image, ratioRange=ZOOM_RANGE, outputNum=1):
    zoomed = []
    for i in range(outputNum):
        ratio = 1 - np.random.uniform(-ratioRange, ratioRange)
        # print(image.shape)
        rescaled = transform.rescale(image, ratio)
        # print(rescaled.shape)
        rows, cols, channel = rescaled.shape
        startTop = np.abs(rows - image.shape[0]) // 2
        startLeft = np.abs(cols - image.shape[1]) // 2
        if ratio > 1:
            result = rescaled[startTop:startTop+image.shape[0],
                              startLeft: startLeft+image.shape[1],
                              :]
        else:
            result = np.zeros(image.shape)
            result[startTop:startTop+rows, startLeft:startLeft+cols, :] = rescaled
        zoomed.append(result)
    if outputNum == 1:
        zoomed = zoomed[0]
    return zoomed


def randomReform(
        image,
        rotateRange=ROTATE_RANGE, shearRange=SHEAR_RANGE, zoomRange=ZOOM_RANGE,
        binarizing=True, mode=MODE, threshold=THRESHOLD,
        outputNum=1
):
    if binarizing:
        binarized = binarize(image, mode, threshold)
    else:
        binarized = image
    result = []
    for count in range(outputNum):
        if zoomRange is not None:
            zoomed = randomZoom(binarized, zoomRange)
        else:
            zoomed = randomZoom(binarized)
        if rotateRange is not None:
            rotated = randomRotate(zoomed, rotateRange)
        else:
            rotated = randomRotate(zoomed)
        if shearRange is not None:
            sheared = randomShear(rotated, shearRange)
        else:
            sheared = randomShear(rotated)
        result.append(sheared)
    if outputNum == 1:
        result = result[0]

    return result
