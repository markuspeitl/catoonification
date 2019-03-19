import regiongrow as regiong

import cv2
import numpy as np

orgimg = cv2.imread('C:/Users/Max/Pictures/53146095_10155809091156428_4847299437430571008_n.jpg')
width,height,ch = orgimg.shape

blurredimg = cv2.bilateralFilter(orgimg,10,30,30)
#blurredimg = cv2.bilateralFilter(orgimg,7,70,70)

resultimg = regiong.growRegionGrid(blurredimg,np.array([50,50]),np.array([10,30,80]))


cv2.imshow('result',resultimg)
cv2.imshow('smallimg',cv2.resize(orgimg,(height,width)))

cv2.waitKey()