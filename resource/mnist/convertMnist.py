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
import numpy as np
from skimage import io
from reform import binarize


def save(labeledImages):
    counter = 0
    for im in labeledImages:
        image, label = im
        io.imsave(arr=image, fname='./train/'+str(label)+'/'+str(counter)+'.png')
        counter += 1


with gzip.open('./mnist.pkl.gz', 'rb') as f:
    data = cPickle.load(f, encoding='bytes')
train, validation, test = data

trainImage, trainLabels = train
trainImage = list(trainImage)
trainLabels = list(trainLabels)
trainImage = [image.reshape(28, 28) for image in trainImage]
trainImage = [binarize(image) for image in trainImage]
train = zip(trainImage, trainLabels)
save(train)

#
# convertMnist.py ends here
