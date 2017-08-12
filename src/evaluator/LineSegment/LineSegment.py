import numpy as np
from skimage import filters, exposure
from evaluator import h2l_debug

debugger = h2l_debug.h2l_debugger


def segment(image):

    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got ' +
                         str(image.shape))

    his = np.dot(image, np.ones(shape=(image.shape[1], 1)))
    value = filters.threshold_otsu(his)
    guessed_seg = his < value

    pre = -1
    index = 0
    start_rows = []
    for x in guessed_seg:
        if x and pre == -1:
            pre = index
        if not x and pre != -1:
            mid = (index + pre) // 2
            start_rows.append(mid)
            pre = -1
        index += 1
    if 0 not in start_rows:
        start_rows.insert(0, 0)
    if image.shape[1] not in start_rows:
        start_rows.append(image.shape[1] - 1)

    extracted_lines = []
    for i in range(len(start_rows) - 1):
        line = image[start_rows[i]: start_rows[i+1], ...]
        if not exposure.is_low_contrast(line):
            extracted_lines.append(line)
    return extracted_lines
