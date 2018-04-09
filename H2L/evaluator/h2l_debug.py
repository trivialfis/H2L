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

from matplotlib import pyplot as plt
import cv2
import os

H2L_DEBUG = False
ORANGE = '\033[38;2;255;185;0m'
RESET = '\033[0m '
DUMP_DIR = '/tmp/h2l/dump'

if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)


class h2l_debugger():
    def plot(self, images, caption=None):
        if H2L_DEBUG:
            if caption is not None:
                print(caption)
            if type(images) is not list and type(images) is not tuple:
                images = [images]
            for img in images:
                plt.title(caption)
                plt.imshow(img, cmap='gray')
                plt.show()

    def save_img(self, image, caption):
        if H2L_DEBUG:
            cv2.imwrite(filename=os.path.join(DUMP_DIR, caption + '.png'),
                        img=image)

    def display(self, *strings):
        if H2L_DEBUG:
            for s in strings:
                print(ORANGE + str(s), end=RESET)
            print('\n', end='')

    def image_info(self, prefix, image):
        print(prefix, '\n',
              '  type : ', ORANGE, type(image), RESET,
              '  dtype: ', ORANGE, image.dtype, RESET,
              '  shape: ', ORANGE, image.shape, RESET)

    def log(self, data):
        if H2L_DEBUG:
            with open('h2l.log', 'a') as f:
                f.write(data)
