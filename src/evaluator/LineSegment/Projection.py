from evaluator.LineSegment import ConstValue
import numpy as np
from skimage import filters

def HorizontalProjection(image):
    width = len(image[0])
    height = len(image)
    hor = np.zeros(height)
    for i in range(height):
        for j in range(width):
            if(image[i][j] == ConstValue.BLACK):
                hor[i] = hor[i] + 1
    return hor

def VerticalProjection(image):
    width = len(image[0])
    height = len(image)
    ver = np.zeros(width)
    for j in range(width):
        for i in range(height):
            if(image[i][j] == ConstValue.BLACK):
                ver[j] = ver[j] + 1
    return ver
