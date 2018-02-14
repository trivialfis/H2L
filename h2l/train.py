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

from .trainer import character_recognizer_cnn
from .trainer import character_recogizer_svm
from .data import characters
from .configuration import characterRecognizerConfig as config
from .evaluator import h2l_debug

import os

h2l_debug.H2L_DEBUG = True


def train_character_cnn(train_path, valid_path):
    data_flow = {
        'train': characters.train_flow(path=train_path),
        'valid': characters.validation_flow(path=valid_path)
    }
    model = character_recognizer_cnn.trainer(data_flow)
    try:
        model.train()
    except KeyboardInterrupt:
        print('\nExit')


def train_character_svm():
    model = character_recogizer_svm.trainer()
    try:
        model.train()
    except KeyboardInterrupt:
        print('\nExit')


def parse_path(path):
    index = path.find(':')
    if index > 0:
        train_path, valid_path = path.split(':')
    else:
        train_path = os.path.abspath(os.path.join(path, config.TRAIN_DATA))
        valid_path = os.path.abspath(
            os.path.join(path, config.VALIDATION_DATA)
        )
        exists = os.path.exists
        if not exists(train_path) or not exists(valid_path):
            raise ValueError(
                'Data path {} or {} not exists:'.format
                (os.path.abspath(train_path), os.path.abspath(valid_path))
            )
    return train_path, valid_path


def train_model(path, model_name):
    train_path, valid_path = parse_path(path)
    if model_name == 'svm':
        train_character_svm()
    elif model_name == 'cnn':
        train_character_cnn(train_path, valid_path)
    else:
        raise ValueError(model_name, ' not known.')


if __name__ == '__main__':
    train_character_cnn()
    # train_character_svm()
