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
# No image enhancement should happen here

import cv2
import os
from normalization import image_utils

SOURCE = '../resource/binarized'
TARGET = '../resource/splited'


def load_images():
    symbols = os.listdir(SOURCE)
    images = {}
    edge = 25
    for sym in symbols:
        image = cv2.imread(os.path.join(SOURCE, sym), 0)
        # image = image_utils.binarize3d(image)
        image[:edge, :] = 0
        image[-edge:, :] = 0
        image[:, :edge] = 0
        image[:, -edge:] = 0
        images[sym] = image
    return images


def split(images):
    characters = {}
    for sym, image in images.items():
        height = image.shape[0] // 12
        width = image.shape[1] // 8
        splited = []
        cur_row = 0
        for i in range(12):
            line = image[cur_row: cur_row + height, :]
            cur_col = 0
            for j in range(8):
                symbol = line[:, cur_col: cur_col + width]
                splited.append(symbol)
                cur_col += width
            cur_row += height
        characters[sym] = splited
    return characters


def save_images(images):
    for sym, ims in images.items():
        os.mkdir(os.path.join(TARGET, sym[:-4]))
        count = 0
        for im in ims:
            filename = os.path.join(TARGET, sym[:-4], str(count) + '.png')
            count += 1
            cv2.imwrite(filename=filename, img=im)


def start():
    images = load_images()
    images = split(images)
    save_images(images)
