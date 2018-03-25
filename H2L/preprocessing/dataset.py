#!/usr/bin/env python3
#
# Copyright © 2018 Fis Trivial <ybbs.daans@hotmail.com>
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

import os
from ..configuration import characterRecognizerConfig as config
from ..normalization import image_utils
from shutil import copyfile
import cv2
import sys
from tqdm import tqdm


def remove_data_comfirm(path):
    confirm = input('Remove old data from '+path+' [y/N]: ')
    if confirm.lower() == 'y':
        return True
    else:
        return False


def split(path):
    os.listdir(os.path.abspath(path))
    pardir = os.path.join(path, os.path.pardir)
    pardir = os.path.normpath(pardir)

    train_dir = os.path.join(pardir, config.TRAIN_DATA)
    valid_dir = os.path.join(pardir, config.VALIDATION_DATA)
    if os.path.exists(train_dir):
        if remove_data_comfirm(train_dir):
            os.system('rm -r ' + train_dir)
        else:
            return -1
    os.mkdir(train_dir)

    if os.path.exists(valid_dir):
        if remove_data_comfirm(valid_dir):
            os.system('rm -r ' + valid_dir)
        else:
            return -1
    os.mkdir(valid_dir)
    symbols = os.listdir(path)

    for sym in symbols:
        sym_path = os.path.join(path, sym)
        images = os.listdir(sym_path)
        dst_train_sym_path = os.path.join(train_dir, sym)
        dst_valid_sym_path = os.path.join(valid_dir, sym)

        if not os.path.exists(dst_train_sym_path):
            os.mkdir(dst_train_sym_path)
        if not os.path.exists(dst_valid_sym_path):
            os.mkdir(dst_valid_sym_path)

        train_size = int(len(images) * config.TV_RATIO)
        train_data = images[:train_size]
        valid_data = images[train_size:]
        for img in train_data:
            src = os.path.join(sym_path, img)
            dst = os.path.join(dst_train_sym_path, img)
            copyfile(src, dst)
        for img in valid_data:
            src = os.path.join(sym_path, img)
            dst = os.path.join(dst_valid_sym_path, img)
            copyfile(src, dst)


def _operation(path, op, outdir):
    os.listdir(os.path.abspath(path))
    pardir = os.path.join(path, os.path.pardir)
    pardir = os.path.normpath(pardir)
    # dir at the same level of input dir.
    output_dir = os.path.join(pardir, outdir)
    if os.path.exists(output_dir):
        if remove_data_comfirm(output_dir):
            os.system('rm -rf ' + output_dir)
        else:
            return -1
    os.mkdir(output_dir)
    filepaths = []
    for root, dirs, files in os.walk(path):
        names = [os.path.join(root, name) for name in files]
        filepaths.extend(names)

    bar = tqdm(total=len(filepaths) // 10, unit='x10 images')
    for i, input_path in enumerate(filepaths):
        # the file path using path as root, e.g. subdirs.
        relpath = os.path.relpath(input_path, path)
        out_path = os.path.join(output_dir, relpath)
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        image = cv2.imread(input_path, 0)
        # image = image_utils.binarize2d_inv(image)
        image = op(image)
        cv2.imwrite(out_path, image)
        if i % 10 == 0:
            bar.update(1)
    bar.close()


def remove_edges(path):
    def _re(image):
        image = image_utils.remove_edges(image, 0.15)
        length = max(image.shape)
        image = image_utils.fill_to_size(image, (length, length))
        return image
    _operation(path, _re, 'noedges')


def binarize(path):
    _operation(path, image_utils.binarize2d_inv, 'binarized')


if __name__ == '__main__':
    path = sys.argv[1]
    split(path)
