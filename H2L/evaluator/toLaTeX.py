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
from ..evaluator import h2l_debug

debugger = h2l_debug.h2l_debugger()

HEAD = "\\documentclass[a4paper, 11pt]{article}\n" + \
       "\\usepackage{amsmath, amssymb}\n" + \
       "\\begin{document}"
TAIL = "\n\end{document}"
EQ_B = "\\begin{equation}\n"
EQ_E = "\n\\end{equation}\n"


def transoform(equations, path='~/Downloads/h2l'):

    outdir = os.path.expanduser(path)
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
    debugger.display('outdir:', outdir)
    try:
        # Parameters must come before input file.
        subprocess.run(
            ['pdflatex', '-output-directory', outdir, outfile],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=10)
    except KeyboardInterrupt:
        print('Interrupted')
    except Exception:
        print("pdflatex command not found. Please install pdflatex",
              " and make sure it's in the system path")
    return outfile[:-3] + 'pdf'
