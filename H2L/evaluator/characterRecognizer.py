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

import sys
from keras import models
import numpy as np
from ..configuration import characterRecognizerConfig as config
from . import h2l_debug

debugger = h2l_debug.h2l_debugger()


class recognizer(object):
    def __init__(self):
        if config.modelExists():
            with open(config.ARCHITECTURE_FILE, 'r') as a:
                self.model = models.model_from_json(a.read())
                self.model.load_weights(config.WEIGHTS_FILE)
                debugger.display('recognizer:\n', config.NAME,
                                 'initialized from file.')
            with open(config.CHARACTER_MAP, 'r') as mapFile:
                mapString = mapFile.read()
                self.charactersMap = eval(mapString)
        else:
            sys.exit('No weight file or json file found')

    def predict_classes(self, x, batch_size=None, verbose=0, steps=None):
        """Generate class predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.

        # Returns
            A numpy array of class predictions.
        """
        proba = self.model.predict(
            x, batch_size=batch_size, verbose=verbose, steps=steps)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def predictCharacter(self, img):
        result = self.model.predict(img, batch_size=1, verbose=False)
        result = self.charactersMap[np.argmax(result)]
        return result

    def predict(self, images):
        if hasattr(self.model, 'predict_classes'):
            class_code = list(
                self.model.predict_classes(
                    images, batch_size=len(images), verbose=False))
        else:
            class_code = list(
                self.predict_classes(
                    images, batch_size=len(images), verbose=False))
        characters = [self.charactersMap[code] for code in class_code]
        return characters

    def predictProbability(self, images):
        probabilities = list(
            self.model.predict(images, batch_size=len(images), verbose=False))
        probabilities = [np.max(probas) for probas in probabilities]
        return probabilities
