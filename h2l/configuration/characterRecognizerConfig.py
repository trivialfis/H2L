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

ARCHITECTURE_FILE = './models/character_recognizer_architure.json'
WEIGHTS_FILE = './models/character_recognizer_weights.hdf5'
VISUAL_FILE = './models/model_plot.png'
NAME = 'character_recognizer'
CHARACTER_MAP = './models/characters_map'

SVM_MODEL = './models/characters_svm.pkl'

# BATCH_SIZE = 32
BATCH_SIZE = 16
# VALIDATION_BATCH_SIZE = 300
VALIDATION_BATCH_SIZE = 64

CLASS_NUM = 36

INIT_LEARNING_RATE = 2.0
# INIT_LEARNING_RATE = 0.02
EPOCH = 3

IMG_ROWS, IMG_COLS = 48, 48
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

# Data directories
TRAIN_DATA = '../resource/training'
VALIDATION_DATA = '../resource/validation'


def modelExists():
    weightsExists = os.path.exists(WEIGHTS_FILE)
    architectureExists = os.path.exists(ARCHITECTURE_FILE)
    return weightsExists and architectureExists


def svm_exists():
    model_exists = os.path.exists(SVM_MODEL)
    return model_exists
