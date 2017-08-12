'''
File:          reform.py
Author:        fis
Created:       Feb 14 2017
Last modified: Aug 12 2017
'''
from skimage import transform, filters, io
import numpy as np
import cv2

MODE = 'less'
THRESHOLD = 'isodata'
ROTATE_RANGE = np.pi / 32
SHEAR_RANGE = np.pi / 32
ZOOM_RANGE = 0.05


def removeEdge(image):
    '''Remove the white edges of the characters'''
    def detectRow(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i if reverse else i
            filled = np.sum(image[index, :])
            if filled == 0.0:
                count += 1
            else:
                break
        return count

    def detectCol(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i if reverse else i
            filled = np.sum(image[:, index])
            if filled == 0.0:
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
        print('top: ', top,
              ' down: ', down,
              ' left: ', left,
              ' right: ', right)
        rows = down - top
        cols = right - left
        if rows < height * 0.1 or cols < width * 0.1:
            print(rows, height*0.2, rows < height*0.2, cols, width*0.2,
                  cols < width*0.2, 'Return full image')
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
        # print('After resize: ', image.shape)
        image = rescale(image, height)
    return image


def padding(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    rows, columns = image.shape
    paddedImage = np.zeros((rows+16, columns+8), dtype=np.uint8)
    paddedImage[4:4+rows, 4:4+columns] = image
    image = resize(paddedImage, outputShape=(rows, columns))
    # image = cv2.threshold(image, 0, 255,
    #                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # image = binarize(image, mode='greater')
    return image


def saveImages(images, prefix=''):
    for im in images:
        saveImages.counter += 1
        io.imsave(arr=im, fname=prefix+str(saveImages.counter)+'.png')


def binarize(image, mode=MODE, threshold=THRESHOLD):
    thMapping = {
        'average': lambda image: np.average(image),
        'isodata': lambda image: filters.threshold_isodata(image),
        'otsu': lambda image: filters.threshold_otsu(image)
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
    maxInput = max(image.shape)
    maxOutput = max(outputShape)
    resized = np.zeros(outputShape)
    if maxInput > maxOutput:
        ratio = maxOutput / maxInput
        resized = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        # resized = transform.rescale(image, maxOutput/maxInput)
    else:
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
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    rotated = []
    for i in range(outputNum):
        angle = np.random.uniform(-angleRange, angleRange)
        degree = angle * 180 / np.pi
        row, col = image.shape
        cy, cx = row // 2, col // 2
        M = cv2.getRotationMatrix2D((cx, cy), degree, 1.0)
        cos_ = np.abs(M[0, 0])
        sin_ = np.abs(M[0, 1])

        new_col = int((row * sin_) + (col * cos_))
        new_row = int((row * cos_) + (col * sin_))

        M[0, 2] += (new_col / 2) - cx
        M[1, 2] += (new_row / 2) - cy
        result = cv2.warpAffine(image, M, (new_col, new_row))
        # result = transform.rotate(image, degree, mode='constant', cval=0,
        #                           resize=False)
        rotated.append(result)
    if outputNum == 1:
        rotated = rotated[0]
    return rotated


def randomShear(image, angleRange=SHEAR_RANGE, outputNum=1):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    sheared = []
    for i in range(outputNum):
        angle = np.random.uniform(-angleRange, angleRange)
        M = np.float32([[1, np.tan(angle), 0],
                        [np.tan(angle), 1, 0]])
        result = cv2.warpAffine(image, M, image.shape)
        # result = transform.warp(image,
        #                         transform.AffineTransform(
        #                             shear=angle),
        #                         mode='constant',
        #                         preserve_range=True)
        sheared.append(result)
    if outputNum == 1:
        sheared = sheared[0]
    return sheared


def randomZoom(image, ratioRange=ZOOM_RANGE, outputNum=1):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    zoomed = []
    for i in range(outputNum):
        ratio = 1 - np.random.uniform(-ratioRange, ratioRange)
        # rescaled = transform.rescale(image, ratio)
        rescaled = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        rows, cols = rescaled.shape
        startTop = np.abs(rows - image.shape[0]) // 2
        startLeft = np.abs(cols - image.shape[1]) // 2
        if ratio > 1:
            result = rescaled[startTop:startTop+image.shape[0],
                              startLeft: startLeft+image.shape[1]]
        else:
            result = np.zeros(image.shape)
            result[startTop:startTop+rows, startLeft:startLeft+cols] = rescaled
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
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    if binarizing:
        binarized = binarize(image, mode, threshold)
    else:
        binarized = image
    binarized = padding(binarized)
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

    return result.astype(np.uint8)
