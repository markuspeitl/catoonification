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

def createSinePath2D(startx,starty,rows,cols):
    height = 100
    nwaves = 20
    pathdist = 200
    anglerad = math.pi/4
    xdir = math.cos(anglerad)
    ydir = math.sin(anglerad)
    print('xdir: ' + str(xdir) + " /ydir: " + str(ydir))
    #direction = [0,1]
    direction = (xdir,ydir)

    #if(xlength > cols - startx):
    #    xlength = cols - startx
    
    linepath = []
    
    #stepy = 2 * math.pi * 1/cols * nwavesy
    
    if abs(xdir) > abs(ydir):
        step = 2 * math.pi * 1/cols * nwaves
        start = startx
        end = int(start + xdir * pathdist)
    else:
        step = 2 * math.pi * 1/cols * nwaves
        start = starty
        end = int(start + ydir * pathdist)

    print("step: " + str(step) + " /start: " + str(start) + " /end: " + str(end))

    currentx = startx
    currenty = starty
    currentdist = 0

    outofbounds = False
    cnt = 0
    while currentdist < pathdist and not outofbounds:
        currentx = startx + cnt * xdir
        currenty = starty + cnt * ydir
        currentdist = math.sqrt(math.pow(currentx - startx,2) + math.pow(currenty - starty,2))
        #print(currentdist)
        x = currentx + int(math.sin(cnt/10) * (height/2) * ydir)
        #y = currenty + int(math.sin(cnt)*height/2) * ydir
        y = currenty + int(math.sin(cnt/10) * (height/2) * xdir)
        if(currentx >= 0 and currentx < cols and currenty >= 0 and currenty < rows):
            linepath.append([int(x),int(y)])
        else:
            outofbounds = True

        cnt += 1

    """for axisit in range(start,end):
        #othermagnitude = int(start + math.sin((axisit-start) * step) * height/2)#  +  direction[1] * x
        #y = starty + int((math.sin((y-starty)*stepy))*(height)/2)
        if abs(xdir) > abs(ydir):
            othermagnitude = int(axisit * xdir)
            linepath.append([axisit,othermagnitude])
        else:
            othermagnitude = int(axisit * ydir)
            linepath.append([othermagnitude,axisit])
    """
    print(linepath)
    return linepath


def drawPathMask(rows,cols,linepath):
    output = np.zeros((rows,cols),np.uint8)

    for point in linepath:
        output[point[1],point[0]] = 255

    cv2.imshow('pathmaskln',output)


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

def walkPath(img,linepath):
    rows, cols, ch = img.shape
    output = np.zeros((rows,cols),np.uint8)

    sigma = 30
    ksizestart = 200
    ksizeend = 40
    startopacity = 0.01
    endopacity = 0.0
    opacitydiff = startopacity - endopacity
    ksizediff = ksizestart - ksizeend

    orgsample = None
    orgGaussK = None
    orgsampleset = False

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
                orgsample = img[startx:endx,starty:endy,:]
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

            kernel = np.ones((currentksize,currentksize),np.uint8) * 255

            lastksize = currentksize

        curropacity = endopacity + opacitydiff * invpercentx
        invopacity = 1 - curropacity

        curropacityK = curropacity * currentgaussK
        invopacityK = 1 - curropacityK
        #curropacityK = currentgaussK
        #invopacityK = 1 - curropacityK
        if(startx >= 0 and endx < rows and starty >= 0 and endy < cols):
            img[startx:endx,starty:endy] = img[startx:endx,starty:endy] * invopacityK + currentsample * curropacityK
            output[startx:endx,starty:endy] = kernel
        else:
            print(rows)



    #for point in linepath:
    #    output[point[1],point[0]] = 255
    cv2.imshow('combimg',img)
    cv2.imshow('pathmask',output)

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
#sinepath = createSinePath(200,400,h,w)
sinepath2d = createSinePath2D(400,300,h,w)
#sinepath = createSinePath(530,120,h,w)
drawPathMask(h,w,sinepath2d)
#drawPathMask(h,w,sinepath)
#walkPath(orgimg,sinepath)
#cv2.imshow('bleed',multiwarp(orgimg,[10,20,30,40,50,60,70,80,90,100],2))
#cv2.imshow('customwarp',customwarp(orgimg))

cv2.waitKey()
cv2.destroyAllWindows()