#!/usr/bin/env python3
#
# Copyright © 2018 Fis Trivial <ybbs.daans@hotmail.com>
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

import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Various commands for H2L.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--split_data', type=str,
        help='''Split a dataset dir into train and valid, new dirs will be
located at the same level as the input dir.
Example:
        h2l_commands --split_data ../data/pngs/'''
    )
    args = parser.parse_args()
    if args.split_data:
        from H2L.preprocessing.split_dataset import split
        split(sys.argv[2])
