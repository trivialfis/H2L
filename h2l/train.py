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

from trainer import characterRecognizerTrainer
from trainer import character_recogizer_svm
from evaluator import h2l_debug
h2l_debug.H2L_DEBUG = True


def trainCharacterRecognizer():
    model = characterRecognizerTrainer.trainer()
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


if __name__ == '__main__':
    trainCharacterRecognizer()
    # train_character_svm()
