# reduce.py ---
#
# Filename: reduce.py
# Description:
# Author: fis
# Maintainer:
# Created: Fri Mar 17 20:21:20 2017 (+0800)
# Version:
# Package-Requires: ()
# Last-Updated: Sat Mar 18 16:14:24 2017 (+0800)
#           By: fis
#     Update #: 50
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
import os
from skimage import io
from random import shuffle


def cleaning(NAME):
    imageFiles = os.listdir('./' + NAME + '/')
    TARGET = './selected' + NAME + '/'
    os.mkdir(TARGET)
    if len(imageFiles) > 10000:
        shuffle(imageFiles)
        imageFiles = imageFiles[:10001]
        path = './' + NAME + '/'
        images = [io.imread(path+img) for img in imageFiles
                  if img.endswith('.png') or img.endswith('.jpg')]
        print(len(images))
        count = 0
        for img in images:
            io.imsave(arr=img, fname=TARGET+str(count)+NAME+'.png')
            count += 1


if __name__ == '__main__':
    for i in range(1, 10):
        print(i)
        cleaning(str(i))
#
# reduce.py ends here
