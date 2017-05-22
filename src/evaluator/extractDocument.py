#coding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawContours(img):
	#img = cv2.imread('../image/test4.jpg',cv2.IMREAD_COLOR)#读入图片
	im = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)#彩色图去噪
	imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)#转化为灰度图
	'''
	gradX = cv2.Sobel(imgray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
	gradY = cv2.Sobel(imgray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)
	blurred = cv2.blur(gradient, (9, 9))
	(_, thresh) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	closed = cv2.erode(closed, None, iterations=4)
	closed = cv2.dilate(closed, None, iterations=4)
	'''
	ret,thresh = cv2.threshold(imgray,150,155,0)
	derp,contour,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#print( type(contour))
	contours = list(contour)
	#print( type(contours))
	#key = cv2.contourArea
	#print( type(key))
	#print(contours)
	c = sorted(contours, key= cv2.contourArea, reverse=True)[0]
	rect = cv2.minAreaRect(c)
	box = np.int0(cv2.boxPoints(rect))
	#cv2.drawContours(im,contours,-1,(0,0,255),3)#画轮廓
	Xs = [i[0] for i in box]
	Ys = [i[1] for i in box]
	x1 = min(Xs)
	x2 = max(Xs)
	y1 = min(Ys)
	y2 = max(Ys)
	hight = y2 - y1
	width = x2 - x1
	cropImg = img[y1:y1+hight, x1:x1+width]
	height = 0.10
	width = 0.10
	#sp[0]为高，sp[1]为宽
	sp = cropImg.shape
	#矫正处理
	sp1 = int(sp[0]*height)
	sp2 = int(sp[1]*width)
	#上左右下
	for num1 in range(0,sp2):
		for num2 in range(0,sp[1]):
			cropImg[num1,num2] = 255
	for num1 in range(0,sp[0]):
		for num2 in range(0,sp2):
			cropImg[num1,num2] = 255
	for num1 in range(0,sp[0]):
		for num2 in range(sp[1]-sp2,sp[1]):
			cropImg[num1,num2] = 255
	for num1 in range(sp[0]-sp2,sp[0]):
		for num2 in range(0,sp[1]):
			cropImg[num1,num2] = 255
	#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	#cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
	#cv2.imshow("image",im)
	#cv2.imshow("Image", cropImg)
	#cv2.waitKey(0)
	#cv2.imwrite(../image/test_after.jpg,img)
	return cropImg
