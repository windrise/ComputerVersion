# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:55:36 2018

@author: 付学明
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

img2 = cv.imread('image/right_03.jpg')
img1 = cv.imread('image/left_03.jpg')

#初始化检测器
orb = cv.ORB_create()

#用ORB寻找关键点和描述子
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None,flags=2)
plt.imshow(img3),plt.show()



'''
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img = cv2.drawKeypoints(gray,kp,img)

cv2.imshow('img',img)
cv2.waitKey(0)

#cv2.imwrite('sift_keypointsright.jpg',img)
'''