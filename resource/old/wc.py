# wc.py --- 
#
# Filename: wc.py
# Description: 
# Author: fis
# Maintainer: 
# Created: Fri Mar 17 20:03:14 2017 (+0800)
# Version: 
# Package-Requires: ()
# Last-Updated: Sat Mar 18 01:40:50 2017 (+0800)
#           By: fis
#     Update #: 35
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
print('='*41)
for root, subdirs, filenames in os.walk('./'):
    subdirs.sort()
    for directory in subdirs:
        images = os.listdir(os.path.join(root, directory))
        print('|\t' + directory + '\t\t|\t' + str(len(images)) + '\t|')
        print('-'*41)
print('='*41)
# 
# wc.py ends here
