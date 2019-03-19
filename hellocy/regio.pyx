import cv2
import numpy as np
cimport numpy as np
np.import_array()

#python setup.py build_ext --inplace
cdef growRegionColMin(np.ndarray hsvimg,np.ndarray sobelimg, pointsToGrow,np.ndarray neigh,np.ndarray resultimg,np.ndarray maxshift,adaptive=True):
        #cdef npy_intp * colw, colh, colch = hsvimg.shape
        cdef int colw = hsvimg.shape[0]
        cdef int colh = hsvimg.shape[1]
        #cdef int colch = hsvimg.shape[2]

        cdef int regionSize = 1
        cdef np.ndarray[np.uint8_t, ndim=1] regionCol = np.empty(3, dtype=np.uint8)

        regionCol = hsvimg[tuple(pointsToGrow[0])]
        print(regionCol)

        cdef np.ndarray[np.uint8_t, ndim=1] neighind = np.empty(2, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1] seedp = np.empty(2, dtype=np.uint8)

        cdef np.ndarray[np.uint8_t, ndim=1] seedval = np.empty(3, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1] neighval = np.empty(3, dtype=np.uint8)
        cdef int sobelval

        print("calling growregion")

        while len(pointsToGrow) > 0:
                seedp = pointsToGrow.pop()
                resultimg[tuple(seedp)] = True

                seedval = hsvimg[tuple(seedp)]
                if adaptive:
                        regionCol = ((regionCol * regionSize + seedval)/(regionSize+1)).astype(np.uint8)
                        regionSize += 1
                        seedval = regionCol

                #seedval = grayimg[tuple(seedp)]
                for shift in neigh:
                        neighind = np.add(seedp ,shift).astype(np.uint8)
                        if neighind[0] in range(0,colw) and neighind[1] in range(0,colh) and not resultimg[tuple(neighind)]:
                                neighval = hsvimg[tuple(neighind)]
                                sobelval = sobelimg[tuple(neighind)]
                                #print(np.absolute(neighval - seedval))
                                #print(maxshift)
                                #print(sobelval)
                                #print(str(all(np.absolute(neighval - seedval) < maxshift)))
                                #print('-------------------------')
                                if sobelval < 50 and all(np.absolute(neighval - seedval) < maxshift):
                                        print('point appended')
                                        pointsToGrow.append(neighind)
        hsvCol = np.uint8([[regionCol]])
        bgrCol = cv2.cvtColor(hsvCol,cv2.COLOR_HSV2BGR)
        return resultimg,bgrCol,regionSize

cpdef growRegionGrid(orgimg,np.ndarray seeddensity,np.ndarray maxshift,adaptive=True,scaledImageRet=False):

        cdef int maxPix = 70000
        orgw, orgh, _ = orgimg.shape

        sobelimg = None

        fullsobel = None

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

        cdef np.ndarray[np.int32_t, ndim=2] neigh = np.empty((1,4), dtype=np.int32)
        neigh = np.array([[1,0],[-1,0],[0,1],[0,-1]])

        if seeddensity[0] > colw:
                seeddensity[0] = colw
        if seeddensity[1] > colh:
                seeddensity[1] = colh

        cdef float stepsizex = colw/seeddensity[0]
        cdef float stepsizey = colh/seeddensity[1]

        minRegionSize = 1

        #allregions = np.zeros((orgw,orgh),dtype=bool)
        allregions = np.zeros((colw,colh),dtype=bool)

        cdef np.ndarray[np.uint8_t, ndim=1] seedpoint = np.empty(2, dtype=np.uint8)
        seedPoints = []

        for x in range(0,seeddensity[0]):
                for y in range(0,seeddensity[1]):

                        seedPoints = []
                        seedpoint[0] = (int)(x*stepsizex)
                        seedpoint[1] = (int)(y*stepsizey)
                        seedPoints.append(seedpoint)

                        if not allregions[tuple(seedpoint)]:
                                #region,regioncol,regionSize = growRegionCol(colimg,seedpoint,maxshift,adaptive=adaptive)
                                region,regioncol,regionSize = growRegionColMin(hsvimg,sobelimg,seedPoints,neigh,falsebuffer.copy(),maxshift,adaptive=True)
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