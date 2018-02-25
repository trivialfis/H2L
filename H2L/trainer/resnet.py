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
from keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPool2D, Layer)
from keras.applications import ResNet50
from keras.engine import InputSpec


class Conv_Block(Layer):
    def __init__(
            self, kernel_size, filters, stage, block, strides=(2, 2),
            **kwargs):
        super(Conv_Block, self).__init__(**kwargs)

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
            ndim=4,
            axes={channel_axis: input_shape[channel_axis]}
        )

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass


class Identity_Block(Layer):

    def __init__(self):
        pass

    def build(self, input_shape):
        pass

    def call(self):
        pass

    def compute_output_shape(self, input_shape):
        pass


def res50(num_classes):
    raise NotImplementedError
