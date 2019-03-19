import cv2
import numpy as np

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
                                if sobelval < 30 and all(np.absolute(neighval - seedval) < maxshift):
                                        pointsToGrow.append(neighind)
        hsvCol = np.uint8([[regionCol]])
        bgrCol = cv2.cvtColor(hsvCol,cv2.COLOR_HSV2BGR)
        return resultimg,bgrCol,regionSize

def growRegionGrid(orgimg,seeddensity,maxshift,adaptive=True,scaledImageRet=False):

        maxPix = 100000
        orgw, orgh, _ = orgimg.shape

        grayimg = cv2.cvtColor(orgimg,cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

        kernely = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
                        
        fullsobel = None

        sobelx = cv2.filter2D(grayimg, -1, kernelx)
        sobely = cv2.filter2D(grayimg, -1, kernely)
        fullsobel = cv2.bitwise_or(sobelx, sobely)

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
        #neigh = [[1,0],[-1,0],[0,1],[0,-1]]

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