import cv2
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('srcimage',help='path of the image to be cartoonified')
parser.add_argument('dstimage',help='path of the image destination')

args = parser.parse_args()

img = None
imgdestpath = None

if(args.srcimage and os.path.isfile(args.srcimage)):
    img = cv2.imread(args.srcimage)
else:
    img = cv2.imread('C:/Users/Max/Pictures/vlcsnap-error808.png')

imgdestpath = args.dstimage

kernelx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

kernely = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

laplacekernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])


blurkernel = np.ones((50,50),np.float32)/2500


#edgeimg = cv2.medianBlur(img,3)
edgeimg = cv2.Canny(img,70,200,3)

#laplaceimg = cv2.filter2D(cv2.medianBlur(img,3), -1, laplacekernel)
laplaceimg = cv2.filter2D(img, -1, laplacekernel)

#edgeimgx = cv2.filter2D(img, -1, kernelx)
#edgeimgy = cv2.filter2D(img, -1, kernely)
#edgeimg = edgeimgx + edgeimgy

morphkernel = np.ones((2,2),np.uint8)
#edgeimg = cv2.erode(edgeimg,morphkernel,iterations = 2)
#edgeimg = cv2.cvtColor(edgeimg,cv2.COLOR_BGR2GRAY)

edgeimg[edgeimg > 240] = 255

#edgeimg = cv2.filter2D(edgeimg, -1, blurkernel)

morphkernel = np.ones((2,2),np.uint8)

#edgeimg = cv2.dilate(edgeimg,morphkernel,iterations = 2)
#edgeimg = cv2.erode(edgeimg,morphkernel,iterations = 2)
#edgeimg = cv2.dilate(edgeimg,morphkernel,iterations = 2)


edgeimg = 255 - edgeimg

#edgeimg = cv2.GaussianBlur(edgeimg,(5,5),0.2,0.2)


#colorimg = img
colorimg1 = cv2.medianBlur(img,7)
colorimg2 = cv2.GaussianBlur(img,(9,9),0.9,0.9)
colorimg3 = cv2.bilateralFilter(img,10,80,80)

colorimg1 = cv2.medianBlur(img,9)
colorimg3 = cv2.bilateralFilter(img,20,120,120)

#colorimg = cv2.medianBlur(img,7)
#colorimg = cv2.GaussianBlur(img,(7,7),0.7,0.7)
#colorimg = cv2.GaussianBlur(img,(5,5),0.5,0.5)
#colorimg = cv2.bilateralFilter(img,6,60,60)
colorimg4 = cv2.addWeighted(colorimg1, 0.5, colorimg2, 0.5, 0.0)
colorimg5 = cv2.addWeighted(colorimg2, 0.5, colorimg3, 0.5, 0.0)
colorimg6 = cv2.addWeighted(colorimg1, 0.5, colorimg3, 0.5, 0.0)

"""cv2.imshow('medianBlur',colorimg1)
cv2.imshow('GaussianBlur',colorimg2)
cv2.imshow('bilateralFilter',colorimg3)
cv2.imshow('medianBlurGaussianBlur',colorimg4)
cv2.imshow('GaussianBlurbilateralFilter',colorimg5)
cv2.imshow('medianBlurbilateralFilter',colorimg6)"""

colorimg = colorimg6

hsv = cv2.cvtColor(colorimg, cv2.COLOR_BGR2HSV)
#hsv[:,:,1] = hsv[:,:,1] * 1.2
#hsv[:,:,0] = 0
#hsv[:,:,2] = hsv[:,:,2] * 0.6
#hsv[:,:,3] = (hsv[:,:,3] - 20) + (hsv[:,:,2]/255)*20
#hsv[hsv < 0] = 0
colorimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#
#colorimg = cv2.medianBlur(img,5)
#colorimg = cv2.medianBlur(img,5)
#colorimg[1] = cv2.medianBlur(img[1],20)
#colorimg[2] = cv2.medianBlur(img[2],20)

#mask = edgeimg/255
#print(mask)

#finalimg = np.multiply(colorimg, mask[:,:,None])
#finalimg = colorimg * mask[:,:,None]
#finalimg = colorimg[:,:,0] * mask
#finalimg[:,:,1] = colorimg[:,:,1] * edgeimg
#finalimg[:,:,2] = colorimg[:,:,2] * edgeimg

#color reduction
factor = 1
colorimg = (colorimg/factor).astype(np.uint8) * factor


#low pass with fft
#grayimg = cv2.cvtColor(cv2.medianBlur(img,3),cv2.COLOR_BGR2GRAY)
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rows, cols = grayimg.shape
crow,ccol = (int)(rows/2) , (int)(cols/2)
dft = cv2.dft(np.float32(grayimg),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)

ratiocol = (int)(cols/3)
ratiorow = (int)(rows/3)
#ratio = (int)(cols/3)

mask = np.zeros((rows,cols,2))
#mask = np.ones((rows,cols,2))
mask[crow-ratiorow:crow+ratiorow, ccol-ratiocol:ccol+ratiocol] = 1
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
cv2.normalize(img_back, img_back, 0, 1, cv2.NORM_MINMAX);
img_back = (img_back * 255).astype(np.uint8)


cannyfft = cv2.Canny(img_back,70,200,3)
#cannyfft = cv2.medianBlur(cannyfft,3)
#cannyfft = cv2.blur(cannyfft,(3,3))
#cannyfft = cv2.dilate(cannyfft,morphkernel,iterations = 1)

finalimglowp = np.copy(colorimg)
finalimglowp[cannyfft > 200] = 0

cannyfft = cv2.GaussianBlur(cannyfft,(3,3),2)
#cv2.imshow('BlurredCanny',cannyfft)

#cannyfft = cannyfft + (cv2.cvtColor(laplaceimg,cv2.COLOR_BGR2GRAY)/255)
#morphkernel = np.ones((2,2),np.uint8)
#cannyfft = cv2.dilate(cannyfft,morphkernel,iterations = 1)

for x in range(0,rows):
    for y in range(0,cols):
        if (cannyfft[x,y] > 30):
            finalimglowp[x,y,:] = finalimglowp[x,y,:]/(cannyfft[x,y]/30)


edgeDet = np.zeros((rows,cols))
edgeDet[cannyfft > 0.1] = 1

ksizeouter = 7
if (ksizeouter % 2) == 0:
    ksizeouter += 1

ksizeinner = ksizeouter - 2
sumkernelouter = np.ones((ksizeouter,ksizeouter))

edgeoffset = 1000

sumkernelinner = np.ones((ksizeinner,ksizeinner))
classPointsIOuter = cv2.filter2D(edgeDet, -1, sumkernelouter)
classPointsInner = cv2.filter2D(edgeDet, -1, sumkernelinner)
classPoints = classPointsIOuter - classPointsInner
#pointImg = np.zeros(img.shape).astype(np.uint8)
pointImg = np.zeros((rows,cols)).astype(np.uint8)
print(pointImg.shape)
print(classPoints.shape)
#pointImg[classPoints == 1] = 255
#pointImg[classPoints == 1] = 255
#pointImg[classPoints > 3] = 1
linestarts = np.logical_and((classPoints == 1), (edgeDet == 1))
middleparts = np.logical_and((classPoints >= 2), (edgeDet == 1))
#truth = (classPoints > 2) & (edgeDet == 1)
pointImg[linestarts == True] = 255
pointImg[middleparts == True] = 60
#print(pointImg)

#pointImg[(classPoints >= 3) & (edgeDet == 1)] = 40




finalimgcanny = np.copy(colorimg)
finalimgcanny[edgeimg == 0] = 0


laplaceimg[laplaceimg > 50] = 255
#laplaceimg = cv2.erode(laplaceimg,morphkernel,iterations = 1)
#laplaceimg = cv2.dilate(laplaceimg,morphkernel,iterations = 3)
laplaceimg = 255 - laplaceimg
finalimglaplace = np.copy(colorimg)
finalimglaplace[laplaceimg == 0] = 0

#opening = cv2.morphologyEx(edgeimg, cv2.MORPH_CLOSE, morphkernel)

"""face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
grayimg2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayimg2, 1.1, 3, 0)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
"""


"""cv2.imshow('org',img)
cv2.imshow('edge',edgeimg)
cv2.imshow('laplace',laplaceimg)
cv2.imshow('col',colorimg)
cv2.imshow('finalcanny',finalimgcanny)
cv2.imshow('finallowp',finalimglowp)
cv2.imshow('finallaplace',finalimglaplace)
#cv2.imshow('finalsobel',finalimglowp)
#cv2.imshow('lowpass',img_back)
cv2.imshow('lowpass',img_back)
cv2.imshow('lowpedges',cannyfft)
cv2.imshow('points',pointImg)"""
#cv2.imshow('lowpass',img_back)
#cv2.imshow('finallowp',finalimglowp)

if(imgdestpath):
    print("writing image to path: " + imgdestpath)
    cv2.imwrite(imgdestpath,finalimglowp)

#cv2.waitKey()

print("finished")

exit(1)