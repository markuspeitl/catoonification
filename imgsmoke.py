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



"""orgimg = orgimg.astype(np.float32)

M = np.float32([[1,0,10],[0,1,0]])

warped = cv2.warpAffine(orgimg,M,(cols,rows))
warped = warped.astype(np.float32)
warped = np.where(warped == 0, orgimg, warped)

output = (warped + orgimg)/2
output = output.astype(np.uint8)

warped = warped.astype(np.uint8)"""
cv2.imshow('newimg',orgimg)
cv2.imshow('bleed',multiwarp(orgimg,[10,20,30,40,50,60,70,80,90,100],2))
cv2.imshow('customwarp',customwarp(orgimg))
#cv2.imshow('bleed',output)

cv2.waitKey()
cv2.destroyAllWindows()