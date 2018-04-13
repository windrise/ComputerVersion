from Stitcher import Stitcher
import cv2

# 读取拼接图片

imageB = cv2.imread("image/left_03.jpg")
imageA = cv2.imread("image/right_03.jpg")

imgA=cv2.resize(imageA,(500,500),interpolation=cv2.INTER_CUBIC)
imgB=cv2.resize(imageB,(500,500),interpolation=cv2.INTER_CUBIC)

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imgA, imgB], showMatches=True)

# 显示所有图片
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()