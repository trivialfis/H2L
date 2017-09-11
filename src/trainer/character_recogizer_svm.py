# character_recogizer_svm.py ---
#
# Filename: character_recogizer_svm.py
# Description:
# Author: fis
# Created: Mon Sep 11 22:59:35 2017 (+0800)
# Last-Updated: Tue Sep 12 01:24:36 2017 (+0800)
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
# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.
#
#

# Code:

from sklearn import svm
import numpy as np
import pickle
from data import characters
from configuration import characterRecognizerConfig as config


class trainer(object):
    def __init__(self):
        self.train_flow = characters.train_flow(1)

        self.images = []
        self.labels = []
        samples = self.train_flow.samples
        count = 0
        for x, y in self.train_flow:
            if count > samples:
                break
            x = x.reshape(x.shape[1:3])
            self.images.append(x.reshape(x.shape[0] * x.shape[1]))
            self.labels.append(np.argmax(y))
            count += 1
        self.images = np.vstack(self.images)
        self.labels = np.vstack(self.labels)
        self.labels = self.labels.reshape((self.labels.shape[0], ))

    def train(self):
        clf = svm.SVC(decision_function_shape='ovo')
        print('Start training SVM.')
        clf.fit(self.images, self.labels)
        with open(config.SVM_MODEL, 'wb') as f:
            pickle.dump(clf, f)

#
# character_recogizer_svm.py ends here
