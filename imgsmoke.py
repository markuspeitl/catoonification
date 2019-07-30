import cv2
import numpy as np
import argparse
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument('dstfilter',help='name of the filter to be executed')
parser.add_argument('srcimage',help='path of the image to be filterd')
parser.add_argument('dstimage',help='path of the image destination')

args = parser.parse_args()

print("drawedges called")
dstfilter = None
srcimage = None
dstimage = None

if(args.dstfilter != None):
    dstfilter = args.dstfilter

if(args.srcimage != None):
    srcimage = args.srcimage

if(args.dstimage != None):
    dstimage = args.dstimage

orgimg = cv2.imread(srcimage)

rows,cols,ch = orgimg.shape

kernel = np.array([ [1,  0,  0,  0,  0],
                    [1,  1,  0,  0,  0],
                    [1,  1,  1,  0,  0],
                    [1,  1,  0,  0,  0],
                    [1,  0,  0,  0,  0]],np.float32)
resized = cv2.resize(orgimg,(int(orgimg.shape[0]/3),int(orgimg.shape[1]/3)))
gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
newimg = cv2.filter2D(gray, -1, kernel/9)

"""factor = 1.1
diff = factor - 1
stretched = cv2.resize(orgimg,(int(orgimg.shape[0]*factor),int(orgimg.shape[1])))
stretched = stretched.astype(np.float32)
orgimg = orgimg.astype(np.float32)
output = (stretched[:,int(orgimg.shape[1]*(diff/2)):int(orgimg.shape[1]*(1+diff/2))] + orgimg)/2
output = output.astype(np.uint8)"""

def multiwarp(img,shifts,orgweight):
    #shifts = [10,20,30,40,50]
    org = img.astype(np.float32)
    output = org.copy()
    for shift in shifts:
        M = np.float32([[1,0,shift],[0,1,0]])
        warped = cv2.warpAffine(org,M,(cols,rows))
        warped = np.where(warped == (0,0,0), org, warped)
        output += warped
    output = (output + org*orgweight)/(len(shifts)+(orgweight+1))
    output = output.astype(np.uint8)

    return output

def customwarp(img):
    output = img.copy()
    cols,rows,ch = img.shape

    """def posfnx(x,y):
        return [int(x*x/16),y]"""

    # 1,2,3
    """def posfnx(x,y,xmax,ymax):
        return [int(math.sin(3*x/xmax) * xmax),y]"""

    """def posfnx(x,y,xmax,ymax):
        return [int(math.sin(5*x/xmax) * xmax),int(math.sin(5*y/ymax) * ymax)]"""

    nwavesx = 1
    nwavesy = 1
    stepx = 2 * math.pi * 1/cols * nwavesx
    stepy = 2 * math.pi * 1/rows * nwavesy 
    def posfnx(x,y,xmax,ymax):
        return [int((math.cos(x*stepx)+1) * xmax/2),int((math.cos(y*stepy)+1) * ymax/2)]

    #https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    #im too lazy to do the maths so copy pasted
    def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    angle = 0.2
    def posfnx(x,y,xmax,ymax):
        newp = rotate((xmax/2,ymax/2),(x,y),angle)
        return [int(newp[0]),int(newp[1])]

    def posfnx(x,y,xmax,ymax):
        angle = math.sqrt(math.pow(xmax-x,2) + math.pow(ymax-y,2))
        newp = rotate((xmax/2,ymax/2),(x,y),angle)
        return [int(newp[0]),int(newp[1])]

    anglescale = 3
    def posfnx(x,y,xmax,ymax):
        angle = anglescale * math.sqrt(math.pow(xmax-x,2) + math.pow(ymax-y,2))*2/(xmax+ymax)
        newp = rotate((xmax/2,ymax/2),(x,y),angle)
        return [int(newp[0]),int(newp[1])]

    anglescale =  math.pi
    def posfnx(x,y,xmax,ymax):
        angle = anglescale * (1-math.sqrt(math.pow(xmax-x,2) + math.pow(ymax-y,2))*2/(xmax+ymax))
        newp = rotate((xmax/2,ymax/2),(x,y),angle)
        return [int(newp[0]),int(newp[1])]

    anglescale =  0.3*math.pi
    def posfnx(x,y,xmax,ymax):
        angle = anglescale * (math.sqrt(math.pow(xmax/2-x,2) + math.pow(ymax/2-y,2))*2/(xmax+ymax))
        newp = rotate((xmax/2,ymax/2),(x,y),angle)
        return [int(newp[0]),int(newp[1])]

    anglescale =  3*math.pi
    def posfnx(x,y,xmax,ymax):
        angle = anglescale * (1-math.sqrt(math.pow(xmax/2-x,2) + math.pow(ymax/2-y,2))*2/(xmax+ymax))
        newp = rotate((xmax/2,ymax/2),(x,y),angle)
        return [int(newp[0]),int(newp[1])]

    """nwaves = 0.8
    def posfnx(x,y,xmax,ymax):
        return [int((math.sin(x*stepx)+1) * xmax/4 + x*0.2),int((math.sin(y*stepy)+1) * ymax/4 + y*0.2)]"""
    
    print(img.shape)
    for x in range(0,cols):
        for y in range(0,rows):
            pos = posfnx(x,y,cols,rows)

            if(pos[1] >= cols):
                pos[1] = cols-1
            if(pos[0] >= rows):
                pos[0] = rows-1

            if(pos[1] < 0):
                pos[1] = 0
            if(pos[0] < 0):
                pos[0] = 0

            #print(pos[1])
            #print(cols)
            output[x,y] = img[pos[1],pos[0]]

    return output

def getMaskBounds(x,y,cols,rows):

    startx = cols/2
    starty = rows/2

    height = 100
    
    startdist = 300
    enddist = 100

    yfactor = y/rows
    xfactor = x/cols

    invyfactor = 1 - yfactor

    nwavesy = 2
    stepy = 2 * math.pi * 1/cols * nwavesy
    
    stepdist = (enddist + startdist * invyfactor)
    higherwidth = startx + stepdist/2 + int((math.sin(y*stepy))*(height)/2)
    lowerwidth = higherwidth - stepdist

    lowerheight = 0
    higherheight = rows

    return [lowerwidth,higherwidth,lowerheight,higherheight]


def createProceduralMask(img):
    cols, rows, ch = img.shape
    centerx = int(rows/2)
    centery = int(cols/2)

    output = np.zeros((cols,rows),np.uint8)

    for x in range(0,cols):
        for y in range(0,rows):

            boundary = getMaskBounds(x,y,cols,rows)
                    
            startx = cols/2
            starty = rows/2

            height = 100
            
            startdist = 300
            enddist = 100

            yfactor = y/rows
            xfactor = x/cols

            invyfactor = 1 - yfactor

            nwavesy = 2
            stepy = 2 * math.pi * 1/cols * nwavesy
            
            stepdist = (enddist + startdist * invyfactor)
            higherwidth = startx + stepdist/2 + int((math.sin(y*stepy))*(height)/2)
            lowerwidth = higherwidth - stepdist

            lowerheight = 0
            higherheight = rows

            boundary =  [lowerwidth,higherwidth,lowerheight,higherheight]    

            if(boundary != None and x > boundary[0] and x < boundary[1] and y > boundary[2] and y < boundary[3]):
                #output[x,y] = int(255 - 255 * (xfactor+yfactor))
                intensity = 0
                if(xfactor < 0.5):
                    intensity = xfactor * 2 * 255
                else:
                    intensity = (1 - xfactor) * 2 * 255 

                output[x,y] = int(intensity)

            #if(y < sinheight):
            #    output[x,y] = 255

    return output

def createSinePath(startx,starty,rows,cols):
    height = 100
    nwaves = 2
    xlength = 800
    direction = [0,1]

    if(xlength > cols - startx):
        xlength = cols - startx
    
    linepath = []
    
    #stepy = 2 * math.pi * 1/cols * nwavesy
    stepx = 2 * math.pi * 1/cols * nwaves

    currentx = startx
    currenty = starty

    for x in range(startx,startx + xlength):
        y = int(starty + math.sin((x-startx) * stepx) * height/2)#  +  direction[1] * x
        #y = starty + int((math.sin((y-starty)*stepy))*(height)/2)
        linepath.append([x,y])

    return linepath

def magnitude(pos1,pos2):
    return math.sqrt(math.pow(pos2[0] - pos1[0],2) + math.pow(pos2[1] - pos1[1],2))

def drawPhysicShot(img):
    imgShape = img.shape

    rows, cols, ch = imgShape
    output = np.zeros((rows,cols),np.uint8)

    seedPoint = np.array([int(cols/2),int(rows/2)])
    seedPoint = seedPoint.astype(np.int32)

    forceDir = np.array([0.1,-1])
    shotForce = 85

    gravityDir = np.array([0,1])
    #gravityConstant = 9.81
    gravityConstant = 4

    distance = 0

    distanceDiff = 0.01
    #distanceDiff = 0.5
    pos = seedPoint
    while(pos[0] >= 0 and pos[0] < cols and pos[1] >= 0 and pos[1] < rows):
        #print(pos)
        output[pos[1],pos[0]] = 255
        #distance = math.sqrt(math.pow(pos[0] - seedPoint[0],2) + math.pow(pos[1] - seedPoint[1],2))
        distance += distanceDiff
        #print("distance: " + str(distance))
        forceDiffComponent = forceDir * shotForce
        gravityDiffComponent = gravityDir * gravityConstant * distance

        #particle velocity in current point
        velocityComponent = forceDiffComponent + gravityDiffComponent

        velocityMagnitude = math.sqrt(math.pow(velocityComponent[0],2) + math.pow(velocityComponent[1],2))
        distanceDiff = 1/(velocityMagnitude)
        print(distanceDiff)

        posChangeComponent = velocityComponent * distance
        pos = (seedPoint + posChangeComponent).astype(np.int32)
        #print(pos)

    cv2.imshow("drawPhysicShot",output)

def createSinePath2D(startx,starty,rows,cols,axisangle):
    startheight = 200
    endheight = 50
    nwaves = 2
    pathdist = 500
    #anglerad = math.pi/3
    anglerad = axisangle
    samplescalef = 1
    xdir = math.cos(anglerad)/samplescalef
    ydir = math.sin(anglerad)/samplescalef
    normxdir = -ydir
    normydir = xdir
    print('xdir: ' + str(xdir) + " /ydir: " + str(ydir))
    startheight = startheight * samplescalef
    endheight = endheight * samplescalef
    heightdiff = endheight - startheight
    #direction = (xdir,ydir)

    linepath = []
 
    #print("step: " + str(step) + " /start: " + str(start) + " /end: " + str(end))

    currentx = startx
    currenty = starty
    currentdist = 0
    startphase = math.pi * 1

    outofbounds = False
    cnt = 0
    sinanglestep = 2 * math.pi * nwaves
    while currentdist < pathdist and not outofbounds:
        percentxy = currentdist/pathdist
        currentx = startx + cnt * xdir
        currenty = starty + cnt * ydir
        currentdist = math.sqrt(math.pow(currentx - startx,2) + math.pow(currenty - starty,2))
        #print(currentdist)
        currheight = startheight + heightdiff * percentxy
        waveheight = math.sin(startphase + percentxy * sinanglestep) * (currheight/2)
        x = currentx + int(waveheight * normxdir)
        y = currenty + int(waveheight * normydir)
        if(currentx >= 0 and currentx < cols and currenty >= 0 and currenty < rows):
            linepath.append([int(x),int(y)])
        else:
            outofbounds = True

        cnt += 1

    #print(linepath)
    return linepath


def drawPathMask(rows,cols,linepath,windowname):
    output = np.zeros((rows,cols),np.uint8)

    for point in linepath:
        output[point[1],point[0]] = 255

    cv2.imshow(windowname,output)


def createCirclularMask(ksize,cthickness=10000):
    if(ksize % 2 is not 0):

        #print('circ ksize:' + str(ksize))
        mask = np.zeros((ksize,ksize),np.uint8)
        centerx = int(ksize/2)
        centery = int(ksize/2)
        radius = int(ksize/2)
        cv2.circle(mask,(centerx,centery),radius,255,thickness=-1)
        if ksize > cthickness*2:
            #print('black circle')
            cv2.circle(mask,(centerx,centery),radius - cthickness,0,thickness=-1)
        #cv2.imshow('mask' + str(ksize),cv2.resize(mask,(300,300)))
        mask.astype(np.float32)
        mask = (mask/255)/(ksize*ksize)
        return mask
    return None

"""def createRandGeometry2D(seedPoint,imgShape):
    output = np.zeros((rows,cols),np.uint8)
"""


def walkPaths(img,orgimg,linepaths):
    rows, cols, ch = img.shape
    output = np.zeros((rows,cols),np.uint8)
    outputmask = np.zeros((rows,cols),np.uint8)
    outimg = np.zeros((rows,cols,3),np.uint8)
    img = img.copy()

    sigma = 30
    ksizestart = 150
    ksizeend = 1
    startopacity = 0.08
    endopacity = 0.0
    filterblackthres = 150
    opacitydiff = startopacity - endopacity
    ksizediff = ksizestart - ksizeend

    orgsample = None
    orgGaussK = None
    orgsampleset = False

    for linepath in linepaths:
        lastksize = -1
        pathlenx = len(linepath)
        for x in range(0,pathlenx):
            percentx = x/pathlenx
            invpercentx = 1 - percentx
            point = linepath[x]
            currentksize = int(ksizeend + ksizediff * invpercentx)

            startx = int(point[1] - currentksize/2)
            endx = int(point[1] + currentksize/2)
            starty = int(point[0] - currentksize/2)
            endy = int(point[0] + currentksize/2)

            if(currentksize != lastksize):
                
                if(not orgsampleset):
                    orgsample = orgimg[startx:endx,starty:endy,:]
                    cv2.imshow('orgsample',orgsample)
                    orgGaussSingleX = cv2.getGaussianKernel(currentksize,sigma) * (sigma/0.4)
                    orgGaussSingleY = cv2.getGaussianKernel(currentksize,sigma) * (sigma/0.4)
                    orgGaussSingle = np.sqrt(orgGaussSingleX * orgGaussSingleY.transpose())
                    #print(orgGaussSingleX)
                    orgGaussK = np.zeros((currentksize,currentksize,3),np.float32)
                    orgGaussK[:,:,0] = orgGaussSingle
                    orgGaussK[:,:,1] = orgGaussSingle
                    orgGaussK[:,:,2] = orgGaussSingle
                    #print(orgGaussSingle)
                    orgsampleset = True

                currentsample = cv2.resize(orgsample,(currentksize,currentksize))
                currentgaussK = cv2.resize(orgGaussK,(currentksize,currentksize))
                #print(currentgaussK)
                #img[startx:endx,starty:endy] = (img[startx:endx,starty:endy] + currentsample)/2

                lastksize = currentksize

            curropacity = endopacity + opacitydiff * invpercentx
            invopacity = 1 - curropacity

            curropacityK = curropacity * currentgaussK
            invopacityK = 1 - curropacityK
            #curropacityK = currentgaussK
            #invopacityK = 1 - curropacityK
            if(startx >= 0 and endx < rows and starty >= 0 and endy < cols):
                if(filterblackthres is not 0):
                    currentgraysample = cv2.cvtColor(currentsample,cv2.COLOR_BGR2GRAY)
                    curropacityK[currentgraysample < filterblackthres] = 0.0
                    invopacityK[currentgraysample < filterblackthres] = 1.0
                img[startx:endx,starty:endy] = img[startx:endx,starty:endy] * invopacityK + currentsample * curropacityK
                #outimg[startx:endx,starty:endy] = img[startx:endx,starty:endy] * invopacityK + currentsample * curropacityK
                output[startx:endx,starty:endy] = currentgaussK[:,:,0] * 255
                outputmask[startx:endx,starty:endy] = 255
            else:
                print(rows)


    #outimg = cv2.bitwise_and(img,img,mask=outputmask)
    #for point in linepath:
    #    output[point[1],point[0]] = 255
    #cv2.imshow('outimg',outimg)
    #cv2.imshow('combimg',img)
    #cv2.imshow('pathmask',output)
    return img

"""orgimg = orgimg.astype(np.float32)

M = np.float32([[1,0,10],[0,1,0]])

warped = cv2.warpAffine(orgimg,M,(cols,rows))
warped = warped.astype(np.float32)
warped = np.where(warped == 0, orgimg, warped)

output = (warped + orgimg)/2
output = output.astype(np.uint8)

warped = warped.astype(np.uint8)"""
cv2.imshow('newimg',orgimg)
#cv2.imshow('mask',createProceduralMask(orgimg))
h, w, ch = orgimg.shape
print('w:' + str(w))
print('h:' + str(h))
sinepath = createSinePath(500,520,h,w)
sinepath2d = createSinePath2D(500,520,h,w,math.pi/3)
#sinepath = createSinePath(530,120,h,w)
drawPathMask(h,w,sinepath2d,'sine2d')
drawPathMask(h,w,sinepath,'sine')
#img1 = walkPath(orgimg,orgimg,sinepath2d)
#img1 = walkPath(img1,orgimg,createSinePath2D(500,520,h,w,0))
img1 = orgimg
linepaths = []
linepaths.append(createSinePath2D(500,520,h,w,math.pi * 2 * -40/360))
linepaths.append(createSinePath2D(500,520,h,w,math.pi * 2 * -20/360))
linepaths.append(createSinePath2D(500,520,h,w,math.pi * 2 * 0/360))
linepaths.append(createSinePath2D(500,520,h,w,math.pi * 2 * 20/360))
linepaths.append(createSinePath2D(500,520,h,w,math.pi * 2 * 40/360))
linepaths.append(createSinePath2D(500,520,h,w,math.pi * 2 * 60/360))
linepaths.append(createSinePath2D(500,520,h,w,math.pi * 2 * 80/360))
#img1 = walkPaths(img1,orgimg,linepaths)
drawPhysicShot(img1)

#img1 = walkPath(img1,orgimg,createSinePath2D(500,520,h,w,math.pi * 2 * -10/360))
cv2.imshow('testorgimgchanged',orgimg)
cv2.imshow('combinedwalked',img1)
#cv2.imshow('bleed',multiwarp(orgimg,[10,20,30,40,50,60,70,80,90,100],2))
#cv2.imshow('customwarp',customwarp(orgimg))

cv2.waitKey()
cv2.destroyAllWindows()