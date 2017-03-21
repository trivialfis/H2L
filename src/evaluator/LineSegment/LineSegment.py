from skimage.draw import line
from skimage import io
import numpy as np
from evaluator.LineSegment import Projection


def segment(image, min_line_height=0, threshold=None):
    hor = Projection.HorizontalProjection(image)
    if(threshold is None):
        threshold = __GetThreshold(hor, 0, len(hor)-1)
    segment = [[0, len(hor)-1]]
    count = 0
    length = len(segment)
    while(count != length):
        count = 0
        length = len(segment)
        line_segment = []
        for i in range(len(segment)):
            # print(segment[i])
            threshold = __GetThreshold(hor, segment[i][0], segment[i][1])
            # print(threshold)
            temp = __PartialSegment(hor, min_line_height, threshold,
                                    segment[i][0], segment[i][1])
            # print(temp)
            if(len(temp) <= 1):
                count = count + 1
            for j in range(len(temp)):
                line_segment.append(temp[j])
        segment = line_segment
    line_pix = [[[]]]
    for i in range(len(segment)):
        temp = __GetRectangle(image, segment[i][0], 0,
                              segment[i][1]-segment[i][0], len(image[0]))
        line_pix.append(temp)
    del line_pix[0]
    line_pix = np.array(line_pix)
    """for i in range(len(line_pix)):
        temp = np.array(line_pix[i])
        line_pix[i] = Transform.Transform(temp)
        io.imshow(line_pix[i])
        io.show()"""
    return line_pix


def __PartialSegment(hor, min_line_height, threshold, begin, end):
    segment = [[]]
    start = begin
    in_line = False
    for i in range(begin, end):
        if(in_line is False and hor[i] > threshold):
            start = i
            in_line = True
        elif(in_line is True and hor[i] < threshold):
            if(i-start >= min_line_height):
                segment.append([start, i])
            in_line = False
    if(in_line is True and end-start >= min_line_height):
        segment.append([start, end])
        in_line = False
    del segment[0]
    return segment


def __GetThreshold(horizontal, begin, end):
    sum = 0
    for i in range(begin, end):
        sum = sum + horizontal[i]
    return (sum/(end-begin+1))*0.1


def __GetRectangle(image, x, y, height, width):
    pixel = [[0 for i in range(width)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            pixel[i][j] = image[i+x][j+y]
    return pixel


def __Testing(image, segment):
    for i in range(len(segment)-1):
        for j in range(segment[i][1], segment[i+1][0]):
            rr, cc = line(j, 0, j, len(image[0])-1)
            image[rr, cc] = 1
    io.imshow(image)
    io.show()
