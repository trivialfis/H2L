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

import numpy as np

from . import image_utils
from ..evaluator import h2l_debug

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
