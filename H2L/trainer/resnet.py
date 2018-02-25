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


class Conv_Block(Layer):

    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 strides=(2, 2),
                 **kwargs):
        super(Conv_Block, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.stage = stage
        self.block = block
        self.strides = strides

    def build(self, input_shape):

        self.build = True
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The chanel dimension of the inputs should be'
                             ' defined, Found `None`.')
        self.input_spec = InputSpec(
            ndim=4, axes={
                channel_axis: input_shape[channel_axis]
            })

    def call(self, input_tensor):

        filters1, filters2, filters3 = self.filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(self.stage) + self.block + '_branch'
        bn_name_base = 'bn' + str(self.stage) + self.block + '_branch'

        x = Conv2D(
            filters1, (1, 1), strides=self.strides,
            name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(
            filters2,
            self.kernel_size,
            padding='same',
            name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(
            filters3, (1, 1), strides=self.strides,
            name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)

    def compute_output_shape(self, input_shape):
        print('input_shape:', input_shape)
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2,
                input_shape[3])


class Identity_Block(Layer):

    def __init__(self, input_tensor, kernel_size, filters, stage, block):
        self.kernel_size = kernel_size
        self.filters = filters
        self.stage = stage
        self.block = block

    def build(self, input_shape):
        self.build = True
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The chanel dimension of the inputs should be'
                             ' defined, Found `None`.')
        self.input_spec = InputSpec(
            ndim=4, axes={
                channel_axis: input_shape[channel_axis]
            })

    def call(self, input_tensor):
        filters1, filters2, filters3 = self.filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(self.stage) + self.block + '_branch'
        bn_name_base = 'bn' + str(self.stage) + self.block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(
            filters2,
            self.kernel_size,
            padding='same',
            name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def res50(num_classes):

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

    x = Conv_Block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = Identity_Block(3, [64, 64, 256], stage=2, block='b')(x)
    x = Identity_Block(3, [64, 64, 256], stage=2, block='c')(x)

    x = Conv_Block(3, [128, 128, 512], stage=3, block='a')(x)
    x = Identity_Block(3, [128, 128, 512], stage=3, block='b')(x)
    x = Identity_Block(3, [128, 128, 512], stage=3, block='c')(x)
    x = Identity_Block(3, [128, 128, 512], stage=3, block='d')(x)

    x = Conv_Block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = Identity_Block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = Identity_Block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = Identity_Block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = Identity_Block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = Identity_Block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = Conv_Block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = Identity_Block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = Identity_Block(3, [512, 512, 2048], stage=5, block='c')(x)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    return model, paras
