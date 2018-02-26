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

import os
from H2L.evaluator import toLaTeX


def get_source():
    fp = os.path.abspath(__file__)
    fp = os.path.join(fp, os.path.pardir)
    fp = os.path.normpath(fp)
    fp = os.path.join(fp, 'resource/test_eqs')
    return fp


def rm_rf(path):
    filename = [
        os.path.join(root, f) for root, subdirs, files in os.walk(path)
        for f in files
    ]
    for name in filename:
        os.remove(name)
    subdirs = [
        os.path.join(root, s) for root, subdirs, files in os.walk(path)
        for s in subdirs
    ]
    for s in subdirs:
        os.rmdir(s)
    os.rmdir(path)


def test_latex():
    source_path = get_source()
    with open(source_path, 'r') as fd:
        source = fd.readlines()
    source = [s.strip('\n') for s in source]
    outdir = os.path.normpath(os.path.join(source_path, os.pardir))
    outdir = os.path.join(outdir, 'temp')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    toLaTeX.transoform(source, outdir)
    assert (os.path.exists(os.path.join(outdir, 'result.tex'))
            and os.path.exists(os.path.join(outdir, 'result.pdf')))
    rm_rf(outdir)
