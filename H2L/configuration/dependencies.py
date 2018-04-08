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

H2L_DEPENDENCIES = (
    (('numpy', 'numpy'), '1.12.1'),
    (('scikit-learn', 'sklearn'), '0.19'),
    (('scikit-image', 'skimage'), '0.12'),
    (('opencv-python', 'cv2'), '3.3.0'),
    (('keras', 'keras'), '2.1.2'),
    (('tensorflow', 'tensorflow'), '1.2.1'),
    (('pydot', 'pydot'), '1.2.0'),
    (('h5py', 'h5py'), '2.7.0'),
    (('tqdm', 'tqdm'), '4.19.5'),
    (('pyparsing', 'pyparsing'), '2.2.0'))


def _construct(dep, index):
    if len(dep) == 1:
        return ((dep[0][0][index], dep[0][1]), )
    return ((dep[0][0][index], dep[0][1]), * _construct(dep[1:], index))


def run_time(dep):
    return _construct(dep, 1)


def build_time(dep):
    return _construct(dep, 0)
