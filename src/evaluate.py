'''
File:          evaluate.py
Author:        fis
Created:       Feb 17 2017
Last modified: Aug 14 2017
'''
from evaluator import heuristicSegmenter
from evaluator import characterRecognizer
from evaluator import toLaTeX
from evaluator import crop_image
from evaluator import line_segmenter

from configuration import characterRecognizerConfig as crconfig
from normalization import image_utils, slope_correct

import numpy as np
import cv2

from evaluator import h2l_debug

import warnings
warnings.filterwarnings('ignore')

debuging = h2l_debug.h2l_debugger()

hs = heuristicSegmenter.segmenter()
cr = characterRecognizer.recognizer()


def build_equation(line):
    def findMid(line):
        '''Find the middle row of image's forground.'''
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(line.shape[0]):
            if np.sum(line[i, :]) >= 1.0:
                top = i
                break
        for i in range(1, line.shape[0]):
            if np.sum(line[-i, :]) >= 1.0:
                bottom = line.shape[0] - i
                break
        middle = (top + bottom) / 2
        return middle

    def isSuper(character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(1, character.shape[0]):
            if np.max(character[-i, :]) > 1.0:
                bottom = character.shape[0] - i
                break
        debuging.display('Middle', middle, 'Bottom', bottom)
        if bottom < middle:
            return True
        else:
            return False

    def isSub(character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y) ,got ' +
                             str(character.shape))
        for i in range(character.shape[0]):
            if np.max(character[i, :]) > 1.0:
                top = i
                # print('Top: ', top)
                break
        if top > middle:
            return True
        else:
            return False

    def is_symbol(character):
        result = len(character) > 2 and character[0] != '^' \
                                                        and character[0] != '_'
        return result

    if len(line.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got ' +
                         str(line.shape))
    characterImages = hs.segment(line)
    characterImages = [im for im in characterImages
                       if not image_utils.is_low_ratio(im)]

    if len(characterImages) == 0:
        debuging.display('Not character found in current line.')
        return

    debuging.display('Got', len(characterImages), 'characters.')
    middle = findMid(line)
    superFlag = [isSuper(char, middle) for char in characterImages]  # exp
    subFlag = [isSub(char, middle) for char in characterImages]  # index
    count = 0
    for c in characterImages:
        debuging.save_img(c, caption='segmented'+str(count))
        count += 1
    characterImages = [image_utils.remove_edges(char)
                       for char in characterImages]
    # count = 0
    # for c in characterImages:
    #     debuging.save_img(c, caption='edge-removed'+str(count))
    #     count += 1
    characterImages = [
        image_utils.fill_to_size(
            char,
            (crconfig.IMG_ROWS, crconfig.IMG_COLS)
        )
        for char in characterImages
    ]
    count = 0
    for c in characterImages:
        debuging.save_img(c, caption='resized'+str(count))
        count += 1
    characterImages = [char.reshape(char.shape+(1, ))
                       for char in characterImages]
    characterImages = np.array(characterImages, dtype=np.float32)
    characters = cr.predict(characterImages)
    debuging.display('Evaluate::build_equation::characters ',
                     characters)
    debuging.display('Evaluate::build_equation::length ',
                     len(characters))
    equation = ''
    for i in range(len(characters)):
        if superFlag[i] and not is_symbol(characters[i]):
            char = '^' + characters[i] + ' '
        elif subFlag[i] and not is_symbol(characters[i]):
            char = '_' + characters[i] + ' '
        else:
            char = characters[i] + ' '
        if is_symbol(char):
            symbol = '\\' + char
        else:
            symbol = char
        equation += symbol
    debuging.display('Evaluate::build_equation equation ',
                     equation)
    return equation


def heursiticGenerate(image):
    "Generate LaTeX pdf from image using heuristic methods."

    if len(image.shape) != 3:
        raise ValueError('Expected image with shape (x, y, z), got ' +
                         str(image.shape))

    image = crop_image.crop_image(image)
    image = image_utils.binarize3d(image)
    image = image_utils.binarize2d_inv(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    debuging.save_img(image, 'binarized')

    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))

    lineImages = line_segmenter.segment(image)
    lineImages = [slope_correct.correct_slope(line) for line in lineImages]
    line_count = 0
    for line in lineImages:
        debuging.save_img(line, 'line_corrected' + str(line_count))
        line_count += 1
    lineImages = [image_utils.rescale_by_height(line, 128)
                  for line in lineImages]
    debuging.display(
        "Evaluate:: Number of line images: ",
        "\033[38;2;255;185;0m" + str(len(lineImages)) + "\033[0m")
    equations = []
    for line in lineImages:
        equations.append(build_equation(line))
    if len(equations) == 0:
        debuging.display('No equation found')
    toLaTeX.transoform(equations)
