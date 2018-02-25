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
from keras.layers import (Conv2D, BatchNormalization, Activation, Layer, Input,
                          MaxPooling2D, AveragePooling2D, Flatten, Dense)
from keras import layers
from keras.models import Model
from keras.engine import InputSpec
import math


class Conv_Block(Layer):

    def __init__(self,
                 kernel_size,
                 filter_length,
                 stage,
                 block,
                 strides=(2, 2),
                 **kwargs):
        super(Conv_Block, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filter_length = filter_length
        self.stage = stage
        self.block = block
        self.strides = strides
        # self.data_format = 'channels_last'
        self.name = 'conv_block_' + str(self.stage) + '_' + self.block

    def build(self, input_shape):

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The chanel dimension of the inputs should be'
                             ' defined, Found `None`.')
        self.input_spec = InputSpec(
            axes={
                channel_axis: input_shape[channel_axis]
            })
        self._built = True

    def call(self, input_tensor):

        # print(self.name, 'input_shape:', input_tensor.shape)
        if K.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(self.stage) + self.block + '_branch'
        bn_name_base = 'bn' + str(self.stage) + self.block + '_branch'

        length = self.filter_length
        x = Conv2D(
            length, self.kernel_size, padding='same', strides=self.strides,
            name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(
            length,
            self.kernel_size,
            padding='same',
            name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        shortcut = Conv2D(
            length, (1, 1), strides=self.strides,
            name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        # print(self.name, 'output_shape:', x.shape)

        return x

    def compute_output_shape(self, input_shape):
        # print(self.name, 'compute_inshape:', input_shape)
        rows = input_shape[1] / 2
        rows = math.ceil(rows)
        cols = input_shape[1] / 2
        cols = math.ceil(cols)
        outshape = (input_shape[0], rows, cols, self.filter_length)
        # print(self.name, 'compute_outshape:', outshape)
        return outshape


class Identity_Block(Layer):

    def __init__(self, kernel_size, filter_length, stage, block, **kwargs):

        self.kernel_size = kernel_size
        self.filter_length = filter_length
        self.stage = stage
        self.block = block
        self.name = 'identity_' + str(self.stage) + '_' + self.block
        super(Identity_Block, self).__init__(**kwargs)

    def build(self, input_shape):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The chanel dimension of the inputs should be'
                             ' defined, Found `None`.')
        self.input_spec = InputSpec(
            axes={
                channel_axis: input_shape[channel_axis]
            })
        self._built = True

    def call(self, input_tensor):

        # print(self.name, 'input_shape:', input_tensor.shape)
        length = self.filter_length
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(self.stage) + self.block + '_branch'
        bn_name_base = 'bn' + str(self.stage) + self.block + '_branch'

        x = Conv2D(
            length,
            self.kernel_size,
            padding='same',
            name=conv_name_base + '2b')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(
            length,
            self.kernel_size,
            padding='same',
            name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        # print(self.name, 'output_shape:', x.shape)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def res32(num_classes):

    paras = {
        'valid_batch_size': 32,
        'batch_size': 16
    }

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

    x = Conv_Block(3, 64, stage=2, block='a')(x)
    x = Identity_Block(3, 64, stage=2, block='b')(x)
    x = Identity_Block(3, 64, stage=2, block='c')(x)

    x = Conv_Block(3, 128, stage=3, block='a')(x)
    x = Identity_Block(3, 128, stage=3, block='b')(x)
    x = Identity_Block(3, 128, stage=3, block='c')(x)
    x = Identity_Block(3, 128, stage=3, block='d')(x)

    x = Conv_Block(3, 256, stage=4, block='a')(x)
    x = Identity_Block(3, 256, stage=4, block='b')(x)
    x = Identity_Block(3, 256, stage=4, block='c')(x)
    x = Identity_Block(3, 256, stage=4, block='d')(x)
    x = Identity_Block(3, 256, stage=4, block='e')(x)

    x = AveragePooling2D(name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(img_input, x, name='resnet50')

    return model, paras
