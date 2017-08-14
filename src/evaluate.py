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
from evaluator.LineSegment import LineSegment

from configuration import characterRecognizerConfig as crconfig
from preprocessing import reform
from normalization import image_utils

import numpy as np

from evaluator import h2l_debug

import warnings
warnings.filterwarnings('ignore')

debuging = h2l_debug.h2l_debugger()

hs = heuristicSegmenter.segmenter()
cr = characterRecognizer.recognizer()


def heursiticGenerate(image):
    "Generate LaTeX pdf from image using heuristic methods."

    def findMid(line):
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(line.shape[0]):
            if np.sum(line[i, :]) >= 1.0:
                top = i
                break
        for i in range(line.shape[0]):
            if np.sum(line[-i, :]) >= 1.0:
                bottom = line.shape[0] - i + 1
                break
        middle = (top + bottom) / 2
        return middle

    def isSuper(character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        for i in range(1, character.shape[0]):
            if np.max(character[-i, :]) != 0:
                bottom = character.shape[0] - i
                break
        if bottom < middle:
            return True
        else:
            return False

    def isSub(character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y) ,got ' +
                             str(character.shape))
        for i in range(character.shape[0]):
            if np.max(character[i, :]) != 0:
                top = i
                # print('Top: ', top)
                break
        if top > middle:
            return True
        else:
            return False

    def segmentCharacters(line):
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))
        characterImages = hs.segment(line)
        middle = findMid(line)
        superFlag = [isSuper(char, middle) for char in characterImages]  # exp
        subFlag = [isSub(char, middle) for char in characterImages]  # index
        count = 0
        for c in characterImages:
            debuging.save_img(c, caption='segmented'+str(count))
            count += 1
        characterImages = [image_utils.remove_edges(char)
                           for char in characterImages]
        count = 0
        for c in characterImages:
            debuging.save_img(c, caption='edge-removed'+str(count))
            count += 1
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
        debuging.display('Evaluate::segmentCharacters::characters ',
                         characters)
        debuging.display('Evaluate::segmentCharacters::length ',
                         len(characters))
        equation = ''
        for i in range(len(characters)):
            if superFlag[i]:
                char = '^' + characters[i] + ' '
                equation += char
                print('^', characters[i], end='')
            elif subFlag[i]:
                char = '_' + characters[i] + ' '
                equation += char
                print('_', characters[i], end='')
            else:
                char = characters[i] + ' '
                equation += char
                print(characters[i], end='')
        print('\nEvaluate::segmentCharacters end\n\n')
        return equation

    if len(image.shape) != 3:
        raise ValueError('Expected image with shape (x, y, z), got ' +
                         str(image.shape))

    image = crop_image.crop_image(image)
    image = image_utils.binarize3d(image)
    image = image_utils.binarize2d_inv(image)
    debuging.save_img(image, 'binarized')
    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))
    lineImages = LineSegment.segment(image)
    debuging.display(type(lineImages), len(lineImages))
    line_count = 0
    for line in lineImages:
        debuging.display('Evaluate:')
        debuging.image_info('line', line)
        debuging.display('Max value: ', np.max(line))
        debuging.save_img(line, 'line_' + str(line_count))
        line_count += 1
    lineImages = [reform.rescale(line, 64) for line in lineImages]
    debuging.display(
        "Evaluate:: line images len: ",
        "\033[38;2;255;185;0m" + str(len(lineImages)) + "\033[0m")
    equations = []
    for line in lineImages:
        equations.append(segmentCharacters(line))
    toLaTeX.transoform(equations)
