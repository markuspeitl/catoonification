#todo contrast and saturation
#restrict taking colors with much different hue
#allow gray for low saturation or high/low brightness

import cv2
import numpy as np
import argparse
import os

orgimg = cv2.imread('C:/Users/Max/Pictures/53146095_10155809091156428_4847299437430571008_n.jpg')
width,height,ch = orgimg.shape

"""kernelx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

kernely = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

laplacekernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])


#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

sobelx = cv2.filter2D(img, -1, kernelx)
sobely = cv2.filter2D(img, -1, kernely)

laplace = cv2.filter2D(img, -1, laplacekernel)

#cv2.imshow('sobelximg',sobelx)
#cv2.imshow('sobelyimg',sobely)
cv2.imshow('sobelimg',cv2.bitwise_or(sobelx, sobely))
cv2.imshow('laplace',laplace)

cv2.imshow('all',cv2.bitwise_or(sobelx, sobely,laplace))"""

#smallimg = cv2.resize(orgimg,(int(height/8), int(width/8)))
#smallimg = cv2.cvtColor(smallimg,cv2.COLOR_BGR2GRAY)


"""def growRegionGrayIter(grayimg,seedp,maxshift,adaptive=True):
        resultimg = np.zeros(grayimg.shape)
        
        neigh = [[1,0],[-1,0],[0,1],[0,-1]]
        maxshift = 40
        grayw, grayh = grayimg.shape

        pointsToGrow = [seedp]
        regionSize = 1
        regionCol = grayimg[tuple(seedp)]

        while len(pointsToGrow) > 0:
                seedp = pointsToGrow.pop()
                resultimg[tuple(seedp)] = 255

                seedval = grayimg[tuple(seedp)]
                if adaptive:
                        regionCol = (regionCol * regionSize + seedval)/(regionSize+1)
                        regionSize += 1
                        seedval = regionCol

                #seedval = grayimg[tuple(seedp)]
                for shift in neigh:
                        neighind = np.add(seedp ,shift)
                        #print(tuple(neighind))
                        if neighind[0] in range(0,grayw) and neighind[1] in range(0,grayh) and resultimg[tuple(neighind)] < 255:
                                #print(neighind)
                                neighval = grayimg[tuple(neighind)]
                                #print(neighval)
                                #print(seedval)
                                if np.absolute(neighval - seedval) < maxshift:
                                        #print('going to neighbor')
                                        pointsToGrow.append(neighind)
        return resultimg,regionCol
"""


grayimg = cv2.cvtColor(orgimg,cv2.COLOR_BGR2GRAY)
kernelx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

kernely = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

sobelx = cv2.filter2D(grayimg, -1, kernelx)
sobely = cv2.filter2D(grayimg, -1, kernely)
fullsobel = cv2.bitwise_or(sobelx, sobely)

#cv2.imshow('sobelimg',cv2.bitwise_or(sobelx, sobely))

def growRegionColMin(hsvimg,sobelimg,pointsToGrow,neigh,resultimg,maxshift,adaptive=True):
        colw, colh, colch = hsvimg.shape
        regionSize = 1
        regionCol = hsvimg[tuple(pointsToGrow[0])]

        while len(pointsToGrow) > 0:
                seedp = pointsToGrow.pop()
                resultimg[tuple(seedp)] = True

                seedval = hsvimg[tuple(seedp)]
                if adaptive:
                        regionCol = (regionCol * regionSize + seedval)/(regionSize+1)
                        regionSize += 1
                        seedval = regionCol

                #seedval = grayimg[tuple(seedp)]
                for shift in neigh:
                        neighind = np.add(seedp ,shift)
                        if neighind[0] in range(0,colw) and neighind[1] in range(0,colh) and not resultimg[tuple(neighind)]:
                                neighval = hsvimg[tuple(neighind)]
                                sobelval = sobelimg[tuple(neighind)]
                                #print(sobelval)
                                if sobelval < 50 and all(np.absolute(neighval - seedval) < maxshift):
                                        pointsToGrow.append(neighind)
        hsvCol = np.uint8([[regionCol]])
        bgrCol = cv2.cvtColor(hsvCol,cv2.COLOR_HSV2BGR)
        return resultimg,bgrCol,regionSize

"""def growRegionCol(colimg,seedp,maxshift,adaptive=True):
        colw, colh, colch = colimg.shape
        #print(colimg.shape)
        resultimg = np.zeros((colw,colh),dtype=bool)
        hsvimg = cv2.cvtColor(colimg,cv2.COLOR_BGR2HSV)
        neigh = [[1,0],[-1,0],[0,1],[0,-1]]
        #maxshift = [40,80,80]

        pointsToGrow = [seedp]
        regionSize = 1
        regionCol = hsvimg[tuple(seedp)]

        while len(pointsToGrow) > 0:
                seedp = pointsToGrow.pop()
                resultimg[tuple(seedp)] = True

                seedval = hsvimg[tuple(seedp)]
                if adaptive:
                        regionCol = (regionCol * regionSize + seedval)/(regionSize+1)
                        regionSize += 1
                        seedval = regionCol

                #seedval = grayimg[tuple(seedp)]
                for shift in neigh:
                        neighind = np.add(seedp ,shift)
                        #print(tuple(neighind))
                        if neighind[0] in range(0,colw) and neighind[1] in range(0,colh) and not resultimg[tuple(neighind)]:
                                #print(neighind)
                                neighval = hsvimg[tuple(neighind)]
                                #print(neighval)
                                #print(seedval)
                                if all(np.absolute(neighval - seedval) < maxshift):
                                        #print('going to neighbor')
                                        pointsToGrow.append(neighind)
        hsvCol = np.uint8([[regionCol]])
        bgrCol = cv2.cvtColor(hsvCol,cv2.COLOR_HSV2BGR)
        return resultimg,bgrCol,regionSize
"""
def growRegionGrid(orgimg,seeddensity,maxshift,adaptive=True,scaledImageRet=False):

        maxPix = 200000
        orgw, orgh, _ = orgimg.shape

        sobelimg = None
        if((orgw * orgh) > maxPix):
                factor = orgw * orgh/maxPix
                colimg = cv2.resize(orgimg,(int(orgh/factor), int(orgw/factor)))

                sobelimg = cv2.resize(fullsobel,(int(orgh/factor), int(orgw/factor)))
        else:
                colimg = orgimg

        colw, colh, colch = colimg.shape

        colresultimg = np.zeros(orgimg.shape).astype(np.uint8)

        falsebuffer = np.zeros((colw,colh),dtype=bool)
        hsvimg = cv2.cvtColor(colimg,cv2.COLOR_BGR2HSV)
        neigh = [[1,0],[-1,0],[0,1],[0,-1],[-1,-1],[1,1],[-1,1],[1,-1]]

        if seeddensity[0] > colw:
                seeddensity[0] = colw
        if seeddensity[1] > colh:
                seeddensity[1] = colh

        stepsizex = colw/seeddensity[0]
        stepsizey = colh/seeddensity[1]

        minRegionSize = 6

        #allregions = np.zeros((orgw,orgh),dtype=bool)
        allregions = np.zeros((colw,colh),dtype=bool)

        for x in range(0,seeddensity[0]):
                for y in range(0,seeddensity[1]):

                        seedpoint = [(int)(x*stepsizex),(int)(y*stepsizey)]
                        if not allregions[tuple(seedpoint)]:
                                #region,regioncol,regionSize = growRegionCol(colimg,seedpoint,maxshift,adaptive=adaptive)
                                region,regioncol,regionSize = growRegionColMin(hsvimg,sobelimg,[seedpoint],neigh,falsebuffer.copy(),maxshift,adaptive=True)
                                if(regionSize >= minRegionSize):
                                        intregion = region.astype(np.uint8) * 255
                                        #print(intregion)

                                        
                                        #intregion = cv2.morphologyEx(intregion, cv2.MORPH_OPEN, np.ones((int(factor*2),int(factor*2)),np.uint8))
                                        #intregion = cv2.morphologyEx(intregion, cv2.MORPH_CLOSE, np.ones((int(factor*2),int(factor*2)),np.uint8))
                                        #intregion = cv2.morphologyEx(intregion, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
                                        #intregion = cv2.morphologyEx(intregion, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8))
                                        #intregion = cv2.resize(intregion,(orgh, orgw),cv2.INTER_LINEAR)
                                        intregion = cv2.resize(intregion,(orgh, orgw),cv2.INTER_LINEAR)
                                        #blurredRegion = cv2.blur(intregion,(int(factor/2),int(factor/2)))
                                        blurredRegion = intregion
                                        #resultimg[intregion > 1] = regioncol
                                        colresultimg[blurredRegion > 0] = regioncol
                                        allregions[region] = True

        cv2.imshow('allregions',cv2.resize(allregions.astype(np.uint8) * 255,(500,500),cv2.INTER_NEAREST))

        return colresultimg


#resultimg,regioncol = growRegionCol(smallimg,[300,100],40)
blurredimg = cv2.bilateralFilter(orgimg,3,30,30)
#blurredimg = cv2.bilateralFilter(orgimg,7,70,70)

resultimg = growRegionGrid(blurredimg,[1000,1000],[18,40,100])

#pixxelart minRegionSize= 1 maxPix = 50000 - 90000, 80-300 seedpoints
#blurredimg = cv2.bilateralFilter(orgimg,3,30,30)
#resultimg = growRegionGrid(blurredimg,80,[15,20,150])

#resultimg,regioncol = growRegionGrayIter(smallimg,[20,20],40)
#smallimg[resultimg] = 0
cv2.imshow('result',resultimg)
#cv2.imshow('result',cv2.resize(resultimg,(height,width)))
cv2.imshow('smallimg',cv2.resize(orgimg,(height,width)))


"""img = orgimg.copy()
width,height,ch = img.shape
res = cv2.resize(img,(int(width/1), int(height/1)))
res = img

#img = cv2.bilateralFilter(img,4,30,30)
img = cv2.bilateralFilter(img,7,50,50)
#img = cv2.medianBlur(img,7)
#img[:,:,0] = cv2.medianBlur(img[:,:,0],5)
#img[:,:,1] = cv2.medianBlur(img[:,:,1],5)
#img[:,:,2] = cv2.medianBlur(img[:,:,2],5)
#img = cv2.GaussianBlur(img,(9,9),0.9,0.9)

bins = 7
hist = cv2.calcHist([img[:,:,0], img[:,:,1], img[:,:,2]], [0, 1, 2], None,[bins, bins, bins], [0, 256, 0, 256, 0, 256])
print(hist)

print(res.shape)
neww,newh,newch = res.shape

step = int(255)/bins

pallette = []

print("creating color palette from histogram")
for x in range(0,bins):
    for y in range(0,bins):
        for z in range(0,bins):
            if(hist[x,y,z] > neww):
                col = [x*step + step/2,y*step + step/2,z*step + step/2]
                colimg = res.copy()
                colimg[:,:,:] = col
                #cv2.imshow('r' + str(x) + str(y) + str(z) ,colimg)
                pallette.append(col)


windowsize = 3
removecols = []
for i in range(0,len(pallette)):
    col = pallette[i]
    if(np.absolute(col[0] - col[1]) < windowsize and np.absolute(col[0] - col[2]) < windowsize):
        print("grey value detected")
        removecols.append(i)

cnt = 0
for i in removecols:
    del pallette[i - cnt]
    cnt+=1

#gray pallette
pallette.append([50,50,50])
pallette.append([100,100,100])
pallette.append([150,150,150])
pallette.append([200,200,200])

#always use pure black and pure white in pallette
pallette.append([0,0,0])
pallette.append([255,255,255])


print(len(pallette))

targetimg = img.copy()
#dist = 4000000
#selectedcolor = [0,0,0]
#lastdistance = 4000000


print('creating color distance matrix (distance from image color to each color of the pallette)')

colddistmatrix = []
for x in range(0,len(pallette)):
    colmatrix = img.copy()
    #colmatrix = colmatrix.astype(np.float16)
    colmatrix = colmatrix.astype(float)
    colmatrix[:,:,:] = pallette[x]
    #print(pallette[x])
    #print(colmatrix[:,:,0])
    #print(colmatrix[:,:,1])
    #print(colmatrix[:,:,2])
    #distmatrix = np.sqrt(np.sum(np.power(img-colmatrix,2),axis=3))
    distmatrix = np.linalg.norm(img-colmatrix,axis=2)
    colddistmatrix.append(distmatrix)
    #print(distmatrix)
    #print('distmatrix appended')

print('getting indizes of pixels with minimum distances in a matrix')

#print(colddistmatrix)
colindexmatrix = np.argmin(colddistmatrix,axis=0)
#print(colindexmatrix)
"""
"""print(pallette[0])
print(colddistmatrix[0][0,0])
print(colddistmatrix[0][9,9])

print(np.linalg.norm(img[0,0]-pallette[0]))
print(np.linalg.norm(img[9,9]-pallette[0]))

xim = np.array([[img[0,0],img[9,9]],[img[0,0],img[9,9]]])
xpa = np.array([[pallette[0],pallette[0]],[pallette[0],pallette[0]]])
print(np.linalg.norm(xim-xpa,axis=2))"""
"""
print('replacing pixels by color values')

for x in range(0,width):
    #print('x cnt: ' + str(x))
    for y in range(0,height):

        ##print(colindexmatrix[x,y])
        targetimg[x,y,:] = pallette[colindexmatrix[x,y]]
"""
"""        lastdistance = 4000000
        selectedcolor = [0,0,0]
        for ind in range(0,len(pallette)):
            #print(ind)
            #print(colddistmatrix[ind][x,y])
            #dist = np.linalg.norm(img[x,y]-pallette[ind])
            #dist = np.sum(np.absolute(col-img[x,y]))

            dist = colddistmatrix[ind][x,y]
            if dist < lastdistance:
                selectedcolor = pallette[ind]
                lastdistance = dist
        targetimg[x,y,:] = selectedcolor
"""

#targetimg = np.take(pallette,colindexmatrix[x,y])
#targetimg[x,y,:] = np.fromfunction(lambda x,y: pallette[colindexmatrix[x,y]], img.shape)


#cv2.imshow('img',targetimg)
#cv2.imshow('img',cv2.resize(targetimg,(500, 500)))

#cv2.imshow('resr',res)


grayimg = cv2.cvtColor(orgimg,cv2.COLOR_BGR2GRAY)
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

#finalimglowp = np.copy(targetimg)
finalimglowp = np.copy(resultimg)
finalimglowp[cannyfft > 200] = 0

cv2.imshow('finalsobel',finalimglowp)


"""factor = 40
floaimg = (res/factor).astype(np.uint8) * factor
cv2.imshow('floaimg',floaimg)"""

"""hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#hsv[:,:,1] = hsv[:,:,1] * 1.2
#hsv[:,:,0] = 0
#hsv[:,:,2] = hsv[:,:,2] * 0.6
#hsv[:,:,3] = (hsv[:,:,3] - 20) + (hsv[:,:,2]/255)*20
#hsv[hsv < 0] = 0

lower0 = np.array([0,0,235])
upper0 = np.array([255,255,255])

lower1 = np.array([0,0,0])
upper1 = np.array([255,255,50])

colorsections = 10
depth = 255
stepsize = depth/colorsections

satresult = 150
valresult = 120

for x in range(1,colorsections):
    lower = np.array([int((x-1)*stepsize),50,100])
    upper = np.array([int((x)*stepsize),200,200])

    mask = cv2.inRange(hsv, lower, upper)

    img[mask == 255] = [int((x)*stepsize - stepsize/2),satresult,valresult]

    cv2.imshow('mask'+str(x),mask)




mask0 = cv2.inRange(hsv, lower0, upper0)
mask1 = cv2.inRange(hsv, lower1, upper1)

#img = cv2.bitwise_and(img,img, mask= mask)

img[mask0 == 255] = 255
img[mask1 == 255] = 0

#img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('maskw',mask0)
cv2.imshow('maskb',mask1)
cv2.imshow('colorimg',img)
"""
cv2.waitKey()