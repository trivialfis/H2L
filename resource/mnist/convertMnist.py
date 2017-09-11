# convertMnist.py ---
#
# Filename: convertMnist.py
# Description:
# Author: fis
# Maintainer: fis
# Created: Tue Mar 14 20:47:02 2017 (+0800)
# Version:
# Package-Requires: ()
# Last-Updated: Sat Mar 18 02:11:23 2017 (+0800)
#           By: fis
#     Update #: 36
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
from six.moves import cPickle
import gzip
import os
from skimage import io, filters
import numpy as np


def save(labeledImages, target):
    counter = 0
    for im in labeledImages:
        image, label = im
        value = filters.threshold_otsu(image)
        mask = image < value
        image = mask.astype(np.uint8)
        image *= 255
        path = os.path.join(target, str(label))
        if not os.path.exists(path):
            os.mkdir(path)
        io.imsave(arr=image, fname=os.path.join(path, str(counter)+'.png'))
        counter += 1


with gzip.open('./mnist.pkl.gz', 'rb') as f:
    data = cPickle.load(f, encoding='bytes')
train, validation, test = data

trainImage, trainLabels = train
trainImage = list(trainImage)
trainLabels = list(trainLabels)
trainImage = [image.reshape(28, 28) for image in trainImage]
train = zip(trainImage, trainLabels)
save(train, 'train')

vali_images, vali_labels = validation
vali_images = list(vali_images)
vali_labels = list(vali_labels)
vali_images = [image.reshape(28, 28) for image in vali_images]
validation = zip(vali_images, vali_labels)
save(validation, 'validation')

test_images, test_labels = test
test_images = list(test_images)
test_labels = list(test_labels)
test_images = [image.reshape(28, 28) for image in test_images]
test = zip(test_images, test_labels)
save(test, 'test')

#
# convertMnist.py ends here
