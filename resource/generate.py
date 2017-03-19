# generate.py --- 
# 
# Filename: generate.py
# Description:
# Author: fis
# Maintainer:
# Created: Thu Mar 16 21:17:16 2017 (+0800)
# Version:
# Package-Requires: ()
# Last-Updated: Sat Mar 18 03:13:04 2017 (+0800)
#           By: fis
#     Update #: 128
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
from reform import randomReform, binarize
from skimage import io
import os

NAME = '0'
TIMES = 1


def generate(imagesPath='./'):
    imagesFilename = [img for root, subdirs, files in os.walk(imagesPath)
                      for img in files]
    images = [io.imread(imagesPath+img) for img in imagesFilename]
    images = [randomReform(img) for img in images]
    count = 0
    combined = images
    # combined = []
    # for l in images:
    #     combined += l
    images = combined
    for img in images:
        io.imsave(arr=binarize(img, mode='less', threshold='average'), fname=imagesPath+NAME+str(count)+'.png')
        count += 1


if __name__ == '__main__':
    # generate('./' + NAME + '/')
    for i in range(1, 10):
        print(i)
        generate('./' + str(i) + '/')
#
# generate.py ends here
