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

import os
import sys


def make_path(relative_path):
    return os.path.normpath(
        os.path.join(os.path.abspath(__file__), os.path.pardir, relative_path))


__h2l_config__ = sys.modules[__name__]


def set_algorithm(algorithm):
    setattr(__h2l_config__, 'ALGORITHM', algorithm)
    setattr(__h2l_config__, 'ARCHITECTURE_FILE',
            make_path('../models/character_' + __h2l_config__.ALGORITHM +
                      '_architure.json'))
    setattr(__h2l_config__, 'WEIGHTS_FILE',
            make_path('../models/character_' + __h2l_config__.ALGORITHM +
                      '_weights.hdf5'))
    setattr(
        __h2l_config__, 'VISUAL_FILE',
        make_path('../models/model_' + __h2l_config__.ALGORITHM + '_plot.png'))


NAME = 'character_recognizer'

CHARACTER_MAP = make_path('../models/characters_map')

SVM_MODEL = make_path('../models/characters_svm.pkl')

BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 64

INIT_LEARNING_RATE = 2.0
EPOCH = 3

IMG_ROWS, IMG_COLS = 48, 48
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

# Data directories
TRAIN_DATA = 'train_data'
VALIDATION_DATA = 'valid_data'

TV_RATIO = 0.8


def modelExists():
    weightsExists = os.path.exists(__h2l_config__.WEIGHTS_FILE)
    architectureExists = os.path.exists(__h2l_config__.ARCHITECTURE_FILE)
    return weightsExists and architectureExists


def svm_exists():
    model_exists = os.path.exists(SVM_MODEL)
    return model_exists
