# wc.py ---
#
# Filename: wc.py
# Author: fis
# Maintainer:
# Created: Fri Mar 17 20:03:14 2017 (+0800)


# Code:
import os
total = 0
print('='*41)
for root, subdirs, filenames in os.walk('./'):
    subdirs.sort()
    for directory in subdirs:
        images = os.listdir(os.path.join(root, directory))
        print('|\t' + directory + '\t\t|\t' + str(len(images)) + '\t|')
        total += len(images)
        print('-'*41)
print('='*41)
print('Total: ', total)
#
# wc.py ends here
