#!/usr/bin/env python3
#
# Copyright © 2017, 2018 Fis Trivial <ybbs.daans@hotmail.com>
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

import unittest
import os
from h2l.configuration import characterRecognizerConfig as config


class test_config(unittest.TestCase):

    def test_make_path(self):
        self.assertEqual(
            config.make_path('../models/characters_map'),
            # .. removes file name only, doesn't cd to upper dir.
            os.path.normpath(
                os.path.join(os.path.abspath(__file__),
                             '../../h2l/models/characters_map')))
