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

from keras.layers import Dense, Dropout, Flatten, merge, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Activation
from keras.regularizers import l2
from keras.models import Model, Sequential
from ..configuration import characterRecognizerConfig as config


def branchModel(num_classes):
    inputLayer = Input(shape=config.INPUT_SHAPE)

    minor = Conv2D(
        16, 5, 5, padding='same',
        activation='relu',
        init='uniform'
    )(inputLayer)
    minor = MaxPooling2D(pool_size=(2, 2))(minor)
    minor = Conv2D(
        32, 3, 3, padding='same',
        activation='relu',
        init='uniform'
    )(minor)
    minor = MaxPooling2D(pool_size=(2, 2))(minor)
    minor = Flatten()(minor)

    large = Conv2D(
        16, 8, 8, padding='same',
        activation='relu',
    )(inputLayer)
    large = MaxPooling2D(pool_size=(8, 8))(large)
    large = Flatten()(large)

    merged = merge([minor, large], mode='concat')
    merged = Dense(1024,
                   W_regularizer=l2(0.01),
                   init='uniform',
                   activity_regularizer=l2(0.01),
                   activation='relu')(merged)
    # merged = Dropout(0.25)(merged)
    merged = Dense(512, activation='relu', init='uniform')(merged)
    merged = Dropout(0.5)(merged)
    outputLayer = Dense(
        num_classes,
        activation='softmax',
        init='uniform',
        kernel_regularizer=l2(0.01),
        activity_regularizer=l2(0.01)
    )(merged)
    model = Model(input=inputLayer, output=outputLayer)
    return model


def sequentialModel(num_classes):

    paras = {'batch_size': 16,
             'valid_batch_size': 64}

    model = Sequential()
    model.add(ZeroPadding2D(
        input_shape=config.INPUT_SHAPE,
        padding=((4, 4), (4, 4))
    ))
    model.add(Conv2D(
        filters=64, kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(0.01),
        # activity_regularizer=l2(0.01)
    ))
    model.add(Conv2D(
        filters=64, kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(0.01),
        # activity_regularizer=l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(
        filters=32, kernel_size=(2, 2),
        padding='valid',
        activation='relu',
        kernel_regularizer=l2(0.01),
        # activity_regularizer=l2(0.01)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,
                    kernel_regularizer=l2(0.01),
                    activity_regularizer=l2(0.01),
                    kernel_initializer='uniform'))
    model.add(Activation('softmax'))
    return model, paras
