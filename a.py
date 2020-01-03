import numpy as np

import cv2

image = cv2.imread('/home/qiyi/下载/timg.jpeg')
image = cv2.resize(image, (100, 100))
image = cv2.resize(image, (1000, 1000))

imageVar=cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow('a', imageVar)
cv2.waitKey()

imageVar = imageVar.var()
print(imageVar)
