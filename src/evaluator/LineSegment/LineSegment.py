import numpy as np
from skimage import filters, exposure
from evaluator import h2l_debug

debugger = h2l_debug.h2l_debugger()


def track(image, start_rows):
    LIMIT = 32
    RANGE = 8
    COL_LENGHT = 64
    paths = []
    for start_row in start_rows:
        path = []
        current_row = start_row
        for col in range(image.shape[1]):
            if current_row - LIMIT - RANGE < 0 or \
               current_row + LIMIT + RANGE >= image.shape[0]:
                path.append(current_row)
                continue
            cost_delta = []
            cost_delta_row = []
            for i in range(-LIMIT, LIMIT):
                considered_row = current_row + i
                piece = image[considered_row - RANGE: considered_row + RANGE,
                              col: col+COL_LENGHT]
                cost = 3 * np.sum(piece)
                cost += np.abs(considered_row - current_row)
                # cost += 2 * np.abs(considered_row - start_row)
                cost_delta.append(cost)
                cost_delta_row.append(considered_row)
            min_index = cost_delta.index(min(cost_delta))
            current_row = cost_delta_row[min_index]
            path.append(current_row)
        paths.append(path)

    if len(paths) == 0:
        return None
    temp = np.copy(image)
    for path in paths:
        col = 0
        for row in path:
            temp[row-2:row+2, col] = 255
            col += 1
    debugger.plot(temp, 'segmented')
    return paths


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

    paths = track(image, start_rows)
    if len(paths) == 0:
        return None

    extracted_lines = []
    for i in range(len(paths) - 1):
        upper_bound = min(paths[i])
        lower_bound = max(paths[i+1])
        distance = lower_bound - upper_bound
        line = np.zeros((distance, image.shape[1]))
        for j in range(len(paths[i]) - 1):
            upper = paths[i][j]
            lower = paths[i+1][j]
            debugger.display('segment: ', upper, lower)
            if lower < upper:
                lower = upper + 2
            delta = upper - upper_bound
            try:
                line[delta:delta+lower-upper, j] = image[upper:lower, j]
            except ValueError:
                debugger.display('segment: path: ', i)
        if not exposure.is_low_contrast(line):
            extracted_lines.append(line)
    debugger.plot(images=extracted_lines)
    return extracted_lines
