import subprocess

HEAD = "\\documentclass[a4paper, 11pt]{article}\n" + \
       "\\usepackage{amsmath, amssymb}\n" + \
       "\\begin{document}"
TAIL = "\end{document}"
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
    subprocess.run(['pdflatex', 'result.tex'], stdout=subprocess.PIPE)
