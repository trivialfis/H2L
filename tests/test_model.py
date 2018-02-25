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

from H2L.trainer import cnns
from H2L.trainer import resnet
from keras.models import Model


def test_build_cnn_sequence():
    model, paras = cnns.sequentialModel(35)
    assert (isinstance(model, Model) and type(paras) is dict
            and len(paras.items()) != 0)


def test_build_cnn_branch():
    model, paras = cnns.sequentialModel(35)
    assert (isinstance(model, Model) and type(paras) is dict
            and len(paras.items()) != 0)


def test_build_res():
    model, paras = resnet.res32(35)
    assert (isinstance(model, Model) and type(paras) is dict
            and len(paras.items()) != 0)
