#!/usr/bin/env python3 -*- coding: utf-8 -*-

import cv2
import numpy as np


def drawContours(img):

        # Reduce noise
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 10, 7, 21)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray
        imgray = cv2.fastNlMeansDenoising(imgray, None, h=2,
                                          templateWindowSize=5,
                                          searchWindowSize=7)
        mask = cv2.adaptiveThreshold(
                imgray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 5, 2)

        derp, contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contour)

        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1

        cropImg = mask[y1:y1+hight, x1:x1+width]
        # height = 0.10
        width = 0.10
        # sp[0]为高，sp[1]为宽
        sp = cropImg.shape
        # 矫正处理
        # sp1 = int(sp[0]*height)
        sp2 = int(sp[1]*width)
        cropImg[0:sp2, 0:sp[1]] = 255
        cropImg[0:sp[0], 0:sp2] = 255
        cropImg[0:sp[0], sp[1]-sp2:sp[1]] = 255
        cropImg[sp[0]-sp2:sp[0], 0:sp[1]] = 255

        return cropImg
