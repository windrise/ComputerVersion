#图片的融合

#cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) → dst.
#其中，alpha 为 src1 透明度，beta 为 src2 透明度.
import cv2
import numpy as np 

img1 = cv2.imread('image/left_01.png')
img2 = cv2.imread('image/left_02.jpg')

img2=cv2.resize(img2,(500,375),interpolation=cv2.INTER_CUBIC)


img_mix = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img_mix',img_mix)
cv2.imwrite('imgMix.jpg',img_mix)
cv2.waitKey(0)
cv2.destroyAllWindows()