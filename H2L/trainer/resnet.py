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

from ..configuration import characterRecognizerConfig as config
import keras.backend as K
from keras.layers import (Conv2D, BatchNormalization, Activation, Input,
                          MaxPooling2D, AveragePooling2D, Flatten, Dense)
from keras.applications import ResNet50
from keras import layers
from keras.models import Model


def identity_block(input_tensor, kernel_size, filter_length, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filter_length, kernel_size, padding='same',
        name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filter_length, kernel_size, padding='same',
        name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filter_length,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filter_length,
        kernel_size,
        padding='same',
        strides=strides,
        name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filter_length, kernel_size, padding='same',
        name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(
        filter_length,
        kernel_size,
        padding='same',
        strides=strides,
        name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def res32(num_classes):

    paras = {'valid_batch_size': 64, 'batch_size': 32}

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    input_shape = config.INPUT_SHAPE
    img_input = Input(shape=input_shape)
    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, 64, stage=2, block='a')
    x = identity_block(x, 3, 64, stage=2, block='b')
    x = identity_block(x, 3, 64, stage=2, block='c')

    x = conv_block(x, 3, 128, stage=3, block='a')
    x = identity_block(x, 3, 128, stage=3, block='b')
    x = identity_block(x, 3, 128, stage=3, block='c')
    x = identity_block(x, 3, 128, stage=3, block='d')

    x = conv_block(x, 3, 256, stage=4, block='a')
    x = identity_block(x, 3, 256, stage=4, block='b')
    x = identity_block(x, 3, 256, stage=4, block='c')
    x = identity_block(x, 3, 256, stage=4, block='d')
    x = identity_block(x, 3, 256, stage=4, block='e')

    x = AveragePooling2D(name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(img_input, x, name='resnet50')

    return model, paras
