# combine_from_mani.py ---
#
# Filename: combine_from_mani.py
# Description:
# Author: fis
# Maintainer:
# Created: Mon Mar 13 00:19:10 2017 (+0800)
# Version:
# Package-Requires: ()
# Last-Updated: Mon Mar 13 11:41:57 2017 (+0800)
#           By: fis
#     Update #: 54
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
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
import pickle
VALI_SIZE = 500
MANIPULATORS_FILE = 'manipulators'
TARGET_FILE = 'characters'

validation = []
for i in range(4):
    filename = MANIPULATORS_FILE + str(i) + '.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    images, labels = data
    validation += list(zip(images[-VALI_SIZE:, ...], labels[-VALI_SIZE:, ...]))
    images = images[:-VALI_SIZE, ...]
    labels = labels[:-VALI_SIZE, ...]
    labeledImages = list(zip(images, labels))
    with open(TARGET_FILE+str(i)+'.pkl', 'wb') as f:
        print('Saving ' + TARGET_FILE+str(i)+'.pkl')
        pickle.dump(labeledImages, f)
        print('Dumped')

with open('validation.pkl', 'wb') as f:
    print('Saving validation.pkl')
    pickle.dump(validation, f)
#
# combine_from_mani.py ends here
