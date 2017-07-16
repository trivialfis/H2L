# generate.py ---
#
# Filename: generate.py
# Description:
# Author: fis
# Maintainer:
# Created: Thu Mar 16 21:17:16 2017 (+0800)
# Version:
# Package-Requires: ()
# Last-Updated: Mon Jul 17 03:08:45 2017 (+0800)
#           By: fis
#     Update #: 131

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
        io.imsave(arr=binarize(img, mode='less', threshold='average'),
                  fname=imagesPath+NAME+str(count)+'.png')
        count += 1


if __name__ == '__main__':
    # generate('./' + NAME + '/')
    for i in range(1, 10):
        print(i)
        generate('./' + str(i) + '/')
#
# generate.py ends here
