import numpy as np
from skimage import filters, exposure
from evaluator import h2l_debug
import cv2

STRIP_RATIO = 0.8

LIMIT = 32
RANGE = 2
COL_LENGHT = 32

debugger = h2l_debug.h2l_debugger()


def track(image, start_rows):
    '''
    Used to find the segmentation curve between lines.
    image: input image
    start_rows: row number of curve starting point
    '''
    paths = []
    for start_row in start_rows:
        path = []
        current_row = start_row
        for col in range(image.shape[1]):
            if current_row - LIMIT - RANGE < 0:
                path.append(0)
                continue
            if current_row + LIMIT + RANGE >= image.shape[0]:
                path.append(image.shape[0]-1)
                continue
            cost_delta = []
            cost_delta_row = []
            for i in range(-LIMIT, LIMIT):
                considered_row = current_row + i
                piece = image[considered_row - RANGE: considered_row + RANGE,
                              col: col+image.shape[1]//5]
                # col:]
                cost = 4 * np.sum(piece)
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
    temp = np.copy(image)       # Used to visualize segmentation curve
    for path in paths:
        col = 0
        for row in path:
            temp[row-1:row+1, col] = 255
            col += 1
    debugger.save_img(temp, 'Segmented image')
    return paths


def extract_images(image, paths):
    '''
    Extract lines from IMAGE with known segmentation PATHS.

    return the extracted images as a list
    '''
    extracted_lines = []
    previous_path = paths[0]
    paths = paths[1:]
    for path in paths:
        upper_bound = min(previous_path)
        lower_bound = max(path)
        height = lower_bound - upper_bound
        if height <= 0:
            continue
        line = np.zeros((height, image.shape[1]))
        status = True
        try:
            for j in range(len(path) - 1):
                upper = previous_path[j]
                lower = path[j]
                distance = lower - upper
                if distance < 0:
                    raise ValueError('Distance between path is negative.',
                                     'lower:', lower,
                                     'upper', upper)
                delta = upper - upper_bound
                try:
                    line[delta:delta+distance, j] = image[upper:lower, j]
                except ValueError as e:
                    debugger.display('Error:', e)
                    debugger.display('delta:', delta, 'dlu:', delta+distance,
                                     'upper:', upper, 'lower:', lower)
                    debugger.display('line shape:',
                                     line[delta:delta+distance].shape,
                                     'image shape:',
                                     image[upper:lower].shape)
                    debugger.display('Ori image shape', image.shape)
        except ValueError as e:
            debugger.display(e)
            continue
        if status:
            if not exposure.is_low_contrast(line):
                extracted_lines.append(line)
            previous_path = path
    return extracted_lines


def padding(image):
    if len(image.shape) != 2:
        raise ValueError('Expected image shape (x, y), got ', image.shape)
    rows, columns = image.shape
    length = 2 * LIMIT + RANGE
    paddedImage = np.zeros((rows+length, columns), dtype=np.uint8)
    paddedImage[length//2:length//2+rows, :] = image
    result = cv2.resize(paddedImage, dsize=(rows, columns),
                        interpolation=cv2.INTER_NEAREST)
    return result


def segment(image):
    '''
    Segment the IMAGE into multiple lines,
    return the lines as a list of images.
    '''
    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got ' +
                         str(image.shape))
    padded = padding(image)
    debugger.save_img(padded, 'padded')
    start_strip = padded[:, :int(padded.shape[1]*STRIP_RATIO)]
    his = np.dot(start_strip,
                 np.ones(shape=(start_strip.shape[1], 1)))
    value = filters.threshold_otsu(his)
    guessed_seg = his < value

    debugger.display('guessed_seg shape:', guessed_seg.shape)
    debugger.display('image shape:', image.shape)
    temp = padded.copy()
    for i in range(len(guessed_seg) - 1):
        if guessed_seg[i]:
            temp[i, :] = 255
    debugger.save_img(temp, 'his')

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
    if padded.shape[0] - 1 not in start_rows:
        start_rows.append(padded.shape[0] - 1)

    debugger.display('start_rows: ', start_rows)
    temp = padded.copy()
    for row in start_rows:
        temp[row, :] = 255
    debugger.save_img(temp, 'start_rows')

    paths = track(padded, start_rows)
    extracted_lines = extract_images(padded, paths)

    return extracted_lines
