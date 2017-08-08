import math
import ConstValue
import numpy as np
from skimage import transform


def Transform(line_pix):
    slope = __GetSlope(line_pix)
    angle = math.atan(slope)
    line_pix = transform.rotate(line_pix, angle, resize=True)
    return line_pix

def __GetSlope(line_pix):
    pix_x = []
    pix_y = []
    for i in range(len(line_pix)):
        for j in range(len(line_pix[i])):
            if(line_pix[i][j] == ConstValue.BLACK):
                pix_x.append(i)
		pix_y.append(j)
    del pix_x[0]
    del pix_y[0]
    slope = __LeastSquare(pix_x, pix_y)
    return slope

def __LeastSquare(pix_x, pix_y):
    length = len(pix_x)
    record = np.zeros(4)
    for i in range(length):
	record[0] += pix_x[i]
	record[1] += pix_x[i] * pix_x[i]
	record[2] += pix_y[i]
    for i,m in enumerate(pix_x):
	    record[3] += m * pix_y[i]
    slope = (record[3] - (record[0] * record[2]) / length) /\
		(record[1] - (record[0] * record[0] / length))
    return slope
