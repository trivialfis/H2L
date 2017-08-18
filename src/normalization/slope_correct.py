import numpy as np
from normalization import image_utils

from evaluator import h2l_debug

debugger = h2l_debug.h2l_debugger()

PADDED = 4

RANGE = np.pi / 16
STEP = 0.05


def padding(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    rows, columns = image.shape
    length = 2 * PADDED
    paddedImage = np.zeros((rows, columns+length), dtype=np.uint8)
    paddedImage[:, length//2:length//2+columns] = image
    return paddedImage


def correct_slope(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got', image.shape)

    ratios = []
    angles = np.arange(-RANGE, RANGE, STEP)

    for angle in angles:
        temp = image.copy()
        temp = image_utils.rotate(temp, angle)
        debugger.save_img(temp, 'rotated')
        temp = image_utils.remove_edges(temp)
        ratio = temp.shape[0] / temp.shape[1]
        ratios.append(ratio)

    least = min(ratios)
    index = ratios.index(least)
    image = image_utils.rotate(image, angles[index])
    image = image_utils.remove_edges(image)
    image = padding(image)
    return image
