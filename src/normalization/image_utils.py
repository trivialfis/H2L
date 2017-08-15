import cv2
from evaluator import h2l_debug
import numpy as np

debugger = h2l_debug.h2l_debugger()


def binarize3d(image):
    '''
    Binarize IMAGE, return the binarized version.
    '''
    # img = cv2.fastNlMeansDenoisingColored(image, None, 3, 10, 7, 21)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray
    imgray = cv2.fastNlMeansDenoising(imgray, None, h=10,
                                      templateWindowSize=7,
                                      searchWindowSize=21)

    mask = cv2.adaptiveThreshold(
        imgray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 5, 2)

    return mask


def binarize2d(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    result = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return result


def binarize2d_inv(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    result = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return result


def rotate(image, angle):
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

    image = cv2.warpAffine(image, M, (new_col, new_row))
    return image


def remove_edges(image):
    '''Remove the white edges of the characters'''
    def detectRow(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i-1 if reverse else i
            filled = np.sum(image[index, :])
            if filled < 1:
                count += 1
            else:
                break
        return count

    def detectCol(image, length, reverse=False):
        count = 0
        for i in range(length):
            index = -i-1 if reverse else i
            filled = np.sum(image[:, index])
            if filled < 1:
                count += 1
            else:
                break
        return count

    def __remove_edges(image):
        height, width = image.shape
        top = detectRow(image, height)
        left = detectCol(image, width)
        down = height - detectRow(image, height, True)
        right = width - detectCol(image, width, True)
        rows = down - top
        cols = right - left
        # debugger.display('left:', left, 'right:', right,
        #                  'top:', top, 'down:', down)
        # debugger.plot(image)
        if rows < height * 0.1 or cols < width * 0.1:
            print('rows: ', rows, 'height * 0.2: ', height*0.2,
                  'rows < height * 0.2: ', rows < height*0.2,
                  'cols: ', cols, 'width * 0.2', width*0.2,
                  'cols < width*0.2: ', cols < width*0.2,
                  'Return full image')
            return image

        result = np.array(image[top: down+1, left: right+1],
                          dtype=np.uint8)
        return result

    if len(image.shape) != 2:
        raise ValueError('Image shape should be (x, y), found' +
                         str(image.shape))
    height, width = image.shape
    image = __remove_edges(image)

    return image


def fill_to_size(image, dsize):
    if len(image.shape) != 2:
        raise ValueError('Image shape should be (x, y), found' +
                         str(image.shape))
    if dsize[0] != dsize[1]:
        raise ValueError('Expected dsize(x, y) x == y')
    rows, cols = image.shape
    length = max(rows, cols)
    if length == rows:
        ratio = dsize[0] / length
    else:
        ratio = dsize[1] / length
    resized = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
    filled = np.zeros(dsize)
    filled_r, filled_c = filled.shape
    delta_r = int(filled_r - resized.shape[0]) // 2
    delta_c = int(filled_c - resized.shape[1]) // 2
    filled[delta_r:delta_r+resized.shape[0],
           delta_c:delta_c+resized.shape[1]] = resized
    return filled
