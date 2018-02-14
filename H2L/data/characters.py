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

import pickle
from random import shuffle
import numpy as np
import os
from keras.utils import np_utils, Sequence
from keras.preprocessing import image as Image
from ..configuration import characterRecognizerConfig as config
from ..evaluator import h2l_debug

debuger = h2l_debug.h2l_debugger()


class symbol_sequence(Sequence):
    def __init__(self):
        self.file_count = 0
        self.batch_count = 0
        self.__load_file__()

    def __len__(self):
        return config.SAMPLES_PER_EPOCH // config.BATCH_SIZE

    def __load_file__(self):
        filename = config.TRAIN_DATA + str(self.file_count) + '.pkl'
        with open(filename, 'rb') as f:
            characters = pickle.load(f)
            shuffle(characters)
            self.images, self.labels = zip(*characters)
            self.length = len(self.labels)
            self.images = np.asarray(self.images, dtype=np.float32)
            self.labels = np.asarray(self.labels, dtype=np.float32)
            self.file_count += 1

    def __getitem__(self, idx):
        if self.file_count == config.FILES_COUNT:
            self.file_count = 0
        if self.batch_count > self.length // config.BATCH_SIZE:
            self.batch_count = 0
            self.__load_file__()

        start = round(config.BATCH_SIZE * self.batch_count)
        end = round(config.BATCH_SIZE * (self.batch_count+1))
        if start >= self.length:
            raise IndexError(
                'Start is greater than the dataset size')
        elif end > self.length:
            batchImages = self.images[start:, ...]
            batchLabels = self.labels[start:, ...]
        else:
            try:
                batchImages = self.images[start: end, ...]
                batchLabels = self.labels[start: end, ...]
            except TypeError:
                # The worker model of keras is not thread safe,
                # the np.asarray might be ignored.
                debuger.display('\ncharacters: Types:\n',
                                'images: ', type(self.images), '\n',
                                'labels: ', type(self.labels), '\n',
                                'start: ',  type(start), '\n',
                                'end: ',    type(end))
                os.abort()

        batchLabels = np_utils.to_categorical(batchLabels,
                                              config.CLASS_NUM)
        self.batch_count += 1
        batch = (batchImages, batchLabels)
        return batch


def validationDataLoader():
    with open(config.VALIDATION_DATA, 'rb') as f:
        validationData = pickle.load(f)
        images, labels = zip(*validationData)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        labels = np_utils.to_categorical(labels, config.CLASS_NUM)
    return (images, labels)


def train_flow(path=config.TRAIN_DATA, batch_size=config.BATCH_SIZE):
    "Generate images flow from training data"
    train_datagen = Image.ImageDataGenerator(
        zca_epsilon=None,
    )
    flow = train_datagen.flow_from_directory(
        path,
        color_mode='grayscale',
        target_size=(config.IMG_ROWS, config.IMG_COLS),
        batch_size=batch_size
    )
    mapping = flow.class_indices
    mapping = dict((v, k) for k, v in mapping.items())
    with open(config.CHARACTER_MAP, 'w') as f:
        f.write(str(mapping))
    return flow


def validation_flow(path=config.VALIDATION_DATA):
    validation_gen = Image.ImageDataGenerator(
        zca_epsilon=None,
    )
    flow = validation_gen.flow_from_directory(
        path,
        color_mode='grayscale',
        target_size=(config.IMG_ROWS, config.IMG_COLS),
        batch_size=config.VALIDATION_BATCH_SIZE
    )
    return flow
