# character_svm.py ---
#
# Filename: character_svm.py
# Description:
# Author: fis
# Created: Tue Sep 12 01:31:02 2017 (+0800)
# Last-Updated: Tue Sep 12 09:56:59 2017 (+0800)
#           By: fis
#
#

# Commentary:
#
#
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#

# Code:

from sklearn.externals import joblib
from configuration import characterRecognizerConfig as config
from evaluator import h2l_debug


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
