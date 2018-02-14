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
from shutil import copyfile
import sys


def split(path):
    os.listdir(os.path.abspath(path))
    pardir = os.path.join(path, os.path.pardir)
    pardir = os.path.normpath(pardir)

    train_dir = os.path.join(pardir, config.TRAIN_DATA)
    valid_dir = os.path.join(pardir, config.VALIDATION_DATA)
    if os.path.exists(train_dir):
        os.system('rm -r ' + train_dir)
    os.mkdir(train_dir)

    if os.path.exists(valid_dir):
        os.system('rm -r ' + valid_dir)
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


if __name__ == '__main__':
    path = sys.argv[1]
    split(path)
