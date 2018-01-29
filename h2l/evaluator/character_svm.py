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

from sklearn.externals import joblib
from ..configuration import characterRecognizerConfig as config
from . import h2l_debug


debugger = h2l_debug.h2l_debugger()


class recognizer(object):
    def __init__(self):
        if config.svm_exists():
            self.classifier = joblib.load(config.SVM_MODEL)
            with open(config.CHARACTER_MAP, 'r') as char_map:
                map_string = char_map.read()
                self.characters_map = eval(map_string)
        else:
            raise ValueError('No trained SVM exist.')

    def predict(self, images):
        # debugger.display('images.shape: ', images.shape)
        samples = list(images.copy())
        # debugger.display('samples len: ', len(samples))
        for i in range(len(samples)):
            samples[i] = samples[i].reshape(
                samples[i].shape[0] * samples[i].shape[1])
        result = self.classifier.predict(samples)
        result = [self.characters_map[code] for code in result]
        debugger.display('predict result: ', result)
        return result


#
# character_svm.py ends here
