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

import subprocess
import os

HEAD = "\\documentclass[a4paper, 11pt]{article}\n" + \
       "\\usepackage{amsmath, amssymb}\n" + \
       "\\begin{document}"
TAIL = "\n\end{document}"
EQ_B = "\\begin{equation}\n"
EQ_E = "\n\\end{equation}\n"


def transoform(equations):

    outdir = os.path.expanduser('~/Downloads/h2l')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = os.path.join(outdir, 'result.tex')
    if os.path.exists(outfile):
        os.remove(outfile)
    f = open(outfile, 'a')
    f.write(HEAD)
    for eq in equations:
        temp = EQ_B + eq + EQ_E
        f.write(temp)
    f.write(TAIL)
    f.close()
    try:
        subprocess.run(['pdflatex', 'result.tex'], stdout=subprocess.PIPE)
    except KeyboardInterrupt:
        print('Interrupted')
    except Exception:
        print("pdflatex command not found. Please install pdflatex",
              " and make sure it's in the system path")
