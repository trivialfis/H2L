import numpy as np
from normalization import image_utils

RANGE = np.pi / 16
STEP = 0.05


def correct_slope(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got', image.shape)

    ratios = []
    angles = np.arange(-RANGE, RANGE, STEP)

    for angle in angles:
        temp = image.copy()
        temp = image_utils.rotate(temp, angle)
        temp = image_utils.remove_edges(temp)
        ratio = temp.shape[0] / temp.shape[1]
        ratios.append(ratio)

    least = min(ratios)
    index = ratios.index(least)
    image = image_utils.rotate(image, angles[index])
    return image
