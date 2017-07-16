"""
Author: fis
Last modified: July 15 2017
Transforming equations string into LaTeX and pdf file.
"""

import subprocess

HEAD = "\\documentclass[a4paper, 11pt]{article}\n" + \
       "\\usepackage{amsmath, amssymb}\n" + \
       "\\begin{document}"
TAIL = "\n\end{document}"
EQ_B = "\\begin{equation}\n"
EQ_E = "\n\\end{equation}\n"


def transoform(equations):
    f = open('result.tex', 'a')
    f.write(HEAD)
    for eq in equations:
        temp = EQ_B + eq + EQ_E
        f.write(temp)
    f.write(TAIL)
    f.close()
    try:
        subprocess.run(['pdflatex', 'result.tex'], stdout=subprocess.PIPE)
    except:
        print("pdflatex command not found. Please install pdflatex",
              " and make sure it's in the system path")
