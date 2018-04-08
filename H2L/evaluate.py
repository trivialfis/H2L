#!/usr/bin/env python3
#
# Copyright © 2017, 2018 Fis Trivial <ybbs.daans@hotmail.com>
#
# This file is part of H2L.
#
# H2L is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# H2L is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with H2L.  If not, see <http://www.gnu.org/licenses/>.
#

from .evaluator import heuristicSegmenter
from .evaluator import characterRecognizer
# from evaluator import character_svm
from .evaluator import toLaTeX
from .evaluator import crop_image
from .evaluator import line_segmenter

from .configuration import characterRecognizerConfig as crconfig
from .normalization import image_utils  # , slope_correct

from .evaluator import h2l_debug

import numpy as np
import cv2


import warnings
warnings.filterwarnings('ignore')

debuging = h2l_debug.h2l_debugger()


class position_finder(object):

    def __init__(self, line, margin=5):
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
        self.global_middle = middle
        self.last_middle = middle
        self.margin = margin

    def set_character_middle(self, character):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(character.shape))
        for i in range(character.shape[0]):
            if np.sum(character[i, :]) >= 1.0:
                top = i
                break
        for i in range(1, character.shape[0]):
            if np.sum(character[-i, :]) >= 1.0:
                bottom = character.shape[0] - i
                break
        middle = (top + bottom) / 2
        self.last_middle = middle

    def is_supper(self, character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(character.shape))
        for i in range(1, character.shape[0]):
            if np.max(character[-i, :]) > 1.0:
                bottom = character.shape[0] - i
                break
        if bottom < middle - self.margin:
            return True
        else:
            return False

    def is_sub(self, character, middle):
        if len(character.shape) != 2:
            raise ValueError('Expected image with shape (x, y) ,got ' +
                             str(character.shape))
        for i in range(character.shape[0]):
            if np.max(character[i, :]) > 1.0:
                top = i
                break
        if top > middle + self.margin:
            return True
        else:
            return False

    def get_positions(self, character_images):
        supper_flags = [self.is_supper(character_images[0], self.last_middle)]
        sub_flags = [self.is_sub(character_images[0], self.last_middle)]
        for i in range(1, len(character_images)):
            if supper_flags[i-1] or sub_flags[i-1]:
                middle = self.global_middle
            else:
                middle = self.last_middle
            sup_flag = self.is_supper(character_images[i], middle)
            sub_flag = self.is_sub(character_images[i], middle)
            supper_flags.append(sup_flag)
            sub_flags.append(sub_flag)
            self.set_character_middle(character_images[i])

        return supper_flags, sub_flags


class equation_builder(object):

    def __init__(self, segmenter, classifier):
        '''
        Build an equation from image to string.

        segmenter: Used to segment line image into characters.
        classfier: Used to recognize a character image.
        '''
        self.segmenter = segmenter
        self.classifier = classifier

    def is_symbol(self, character):
            result = (len(character) > 2 and
                      character[0] != '^' and
                      character[0] != '_')
            return result

    def build(self, line):
        '''
        Build the equation line.
        line: Input line image.

        Return: A string representing the equation.
        '''
        if len(line.shape) != 2:
            raise ValueError('Expected image with shape (x, y), got ' +
                             str(line.shape))

        characterImages = self.segmenter.segment(line)
        characterImages = [im for im in characterImages
                           if not image_utils.is_low_ratio(im)]

        if len(characterImages) == 0:
            debuging.display('Not character found in current line.')
            return

        debuging.display('Got', len(characterImages), 'characters.')

        positioner = position_finder(line)
        superFlag, subFlag = positioner.get_positions(characterImages)
        count = 0
        for c in characterImages:
            debuging.save_img(c, caption='segmented'+str(count))
            count += 1
        characterImages = [image_utils.remove_edges(char)
                           for char in characterImages]
        kernel = np.ones((2, 2), np.uint8)
        characterImages = [cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                           for img in characterImages]
        characterImages = [cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                           for img in characterImages]
        count = 0
        for c in characterImages:
            debuging.save_img(c, caption='morphologyEx'+str(count))
            count += 1
        # kernel = np.ones((2, 2), np.uint8)
        # characterImages = [cv2.erode(char, kernel, iterations=1)
        #                    for char in characterImages]
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
        characters = self.classifier.predict(characterImages)
        debuging.display('Evaluate::build_equation::characters ',
                         characters)
        debuging.display('Evaluate::build_equation::length ',
                         len(characters))
        equation = ''
        for i in range(len(characters)):
            if superFlag[i] and not self.is_symbol(characters[i]):
                char = '^' + characters[i] + ' '
            elif (subFlag[i] and
                  not self.is_symbol(characters[i]) and
                  not characters[i] in [',', '-']):
                char = '_' + characters[i] + ' '
            else:
                char = characters[i] + ' '
            if self.is_symbol(char):
                symbol = '\\' + char
            else:
                symbol = char
            equation += symbol
        debuging.display('Evaluate::build_equation equation ',
                         equation)
        return equation


def heursiticGenerate(image, bar=None):
    "Generate LaTeX pdf from image using heuristic methods."

    if len(image.shape) != 3:
        raise ValueError('Expected image with shape (x, y, z), got ' +
                         str(image.shape))

    image = crop_image.crop_image(image)
    image = image_utils.binarize3d(image)
    image = image_utils.binarize2d_inv(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    debuging.save_img(image, 'binarized')

    if len(image.shape) != 2:
        raise ValueError('Expected image with shape (x, y), got '
                         + str(image.shape))

    lineImages = line_segmenter.segment(image)
    # lineImages = [slope_correct.correct_slope(line) for line in lineImages]
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

    hs = heuristicSegmenter.segmenter()
    cr = characterRecognizer.recognizer()
    builder = equation_builder(hs, cr)

    if bar is not None:
        total = len(lineImages)
        count = 0
    for line in lineImages:
        equations.append(builder.build(line))
        if bar is not None:
            count += 1
            bar.set_fraction(count / total)

    if len(equations) == 0:
        debuging.display('No equation found')

    outfile = toLaTeX.transoform(equations)
    return outfile

#
# evaluate.py ends here
