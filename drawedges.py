import cv2
import numpy as np
import argparse
import os

#util
def applyimgfilter(img,applyfilter):
    #print("lowpassfilter")
    height = img.shape[0]
    width = img.shape[1]
    channels = 1
    imgchannels = []
    if(len(img.shape) > 2):
        channels = img.shape[2]
        imgchannels = cv2.split(img)
    else:
        imgchannels = [img]

    for i in range(0,len(imgchannels)):
        imgchannels[i] = applyfilter(imgchannels[i],(height,width,channels))

    if(channels > 1):
        return cv2.merge(imgchannels)
    else:
        return imgchannels[0]

#specific
def fourierfilter(img,shape,mask):
    #apply discrete fourier transform
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    #shifts zero-frequency component to center of spectrum
    dft_shift = np.fft.fftshift(dft)

    #cv2.imshow('spectrum0',dft_shift[:,:,0])
    #cv2.imshow('spectrum1',dft_shift[:,:,1])

    #apply mask to remove low frequency component (only mask widow values are left alone)
    dft_shift = dft_shift * mask
    #dft_shift = cv2.bitwise_and(dft_shift,mask)
    #unshift spectrum
    dft = np.fft.ifftshift(dft_shift)
    #inverse the fourier transform
    img_back = cv2.idft(dft)

    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 1, cv2.NORM_MINMAX)
    img_back = (img_back * 255).astype(np.uint8)

    #cv2.imshow('img_back',img_back)

    return img_back

def sobelFilter(img):
    #print("sobelFilter")
    img.astype(np.uint8)
    grad_x = cv2.Sobel(img,cv2.CV_16U,1,0,ksize=3)
    grad_y = cv2.Sobel(img,cv2.CV_16U,0,1,ksize=3)
    dstimg = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5,0)
    scaled = cv2.convertScaleAbs(dstimg)
    #cv2.imshow('dstimg',scaled)
    return scaled

def applysobelfilter(img):
    return applyimgfilter(img,lambda imgchannel,shape: sobelFilter(imgchannel))

def applybandpassfilterpercent(img,highppercent,lowppercentx):
    shape = img.shape
    return applybandpassfilter(img,highppercent * shape[0]/200,highppercent * shape[1]/200,lowppercentx * shape[0]/200,lowppercentx * shape[1]/200)

def applybandpassfilter(img,highpassy,highpassx,lowpassy,lowpassx):
    
    shape = img.shape
    windowheight = shape[0]/10
    windowwidth = shape[1]/10
    
    centery, centerx = (shape[0]/2) , (shape[1]/2)
    
    mask1 = np.ones((shape[0],shape[1],2),np.uint8)*255
    mask2 = np.zeros((shape[0],shape[1],2),np.uint8)

    """if highpassy > 1:
        drawmask = np.zeros((shape[0],shape[1],2),np.uint8)
        drawmask = cv2.circle(drawmask,(int(centerx),int(centery)),int(highpassy*2),color=(255,255),thickness=-1)
        mask1[drawmask == 255] = 0

    if lowpassy > 1:
        drawmask2 = np.zeros((shape[0],shape[1],2),np.uint8) 
        drawmask2 = cv2.circle(drawmask2,(int(centerx),int(centery)),int(lowpassy*2),color=(255,255),thickness=-1)
        mask2[drawmask2 == 255] = 255
    """

    #highp
    mask1[int(centery - highpassy):int(centery + highpassy), 
        int(centerx - highpassx):int(centerx + highpassx)] = 0

    #lowpass
    mask2[int(centery - lowpassy):int(centery + lowpassy), 
        int(centerx - lowpassx):int(centerx + lowpassx)] = 1

    mask = cv2.bitwise_and(mask2,mask1)
    #cv2.imshow('mask1',mask[:,:,0])
    #cv2.imshow('mask2',mask[:,:,1])

    return applyimgfilter(img,lambda imgchannel,shape: fourierfilter(imgchannel,shape,mask))

def directionglow(img):
    shape = img.shape
    
    centery, centerx = (shape[0]/2) , (shape[1]/2)
    mask2 = np.zeros((shape[0],shape[1],2),np.uint8)
    #20,40
    mask2[int(centery + shape[0]/16 - 30):int(centery+ shape[0]/16 + 30), :] = 255

    return applyimgfilter(img,lambda imgchannel,shape: fourierfilter(imgchannel,shape,mask2))

def trippyfilter(img):
    shape = img.shape
    centery, centerx = (shape[0]/2) , (shape[1]/2)

    mask2 = np.zeros((shape[0],shape[1],2),np.uint8)
    mask1 = np.zeros((shape[0],shape[1],2),np.uint8)
    #mask1 = cv2.circle(mask1,(int(centery),int(centerx)),int(0),color=(255),thickness=-1)
    mask1 = cv2.circle(mask1,(int(centerx),int(centery)),int(0),color=(255),thickness=-1)
    mask1 = 255 - mask1

    #or 100
    #mask2 = cv2.circle(mask2,(int(centery),int(centerx)),int(40),color=(255),thickness=-1)
    mask2 = cv2.circle(mask2,(int(centerx),int(centery)),int(40),color=(255),thickness=-1)

    mask = cv2.bitwise_and(mask1,mask2)
    return applyimgfilter(img,lambda imgchannel,shape: fourierfilter(imgchannel,shape,mask))

def trippyfilter2(img,phase):
    shape = img.shape
    windowheight = shape[0]/10
    windowwidth = shape[1]/10
    
    centery, centerx = (shape[0]/2) , (shape[1]/2)

    mask2 = np.zeros((shape[0],shape[1],2),np.uint8)
    mask1 = np.zeros((shape[0],shape[1],2),np.uint8)
    mask1 = cv2.circle(mask1,(int(centerx),int(centery)),int(0),color=(0,255),thickness=-1)
    mask1 = 255 - mask1

    mask2 = cv2.circle(mask2,(int(centerx),int(centery)),int(100*2)*shape[1],color=(0,255),thickness=-1)

    mask = cv2.bitwise_and(mask1,mask2)

    #mask1res = np.where(mask[:,:,0] != 0)
    #print(mask1res)
    mask2res = np.where(mask[:,:,1] != 255)
    print(mask2res)
    print(mask[369,640,1])

    mask[:,:,0] = 0
    #mask[int(mask.shape[0]/2),int(mask.shape[1]/2),0] = 0
    #mask[:,:,1] = 255
    mask[int(mask.shape[0]/2),int(mask.shape[1]/2),1] = phase

    #cv2.imshow('mask1',mask[:,:,0])
    #cv2.imshow('mask2',mask[:,:,1])

    return applyimgfilter(img,lambda imgchannel,shape: fourierfilter(imgchannel,shape,mask))

def mosaicfilter(img):
    return applybandpassfilterpercent(img,5,15)

def netfilter(img):
    return applybandpassfilterpercent(img,15,30)

def noiserfilter(img):
    return applybandpassfilterpercent(img,40,60)

def colblob(img):
    return applybandpassfilterpercent(img,0,1)

def formsurfaceblobs(img):
    return applybandpassfilterpercent(img,1,5)

def smallformsurfaceblobs(img):
    return applybandpassfilterpercent(img,5,11)

def smallsurfaceblobs(img):
    return applybandpassfilterpercent(img,3,5)

def defaultlowp(img):
    return applybandpassfilterpercent(img,0,30)

def defaulthighp(img):
    return applybandpassfilterpercent(img,50,100)


def getEdges(img):
    lowfreqimg = applybandpassfilterpercent(img,0,30)
    return cv2.Canny(lowfreqimg,100,180,3)

def getEdges2(img):
    lowfreqimg = applybandpassfilterpercent(img,0,45)
    return cv2.Canny(lowfreqimg,70,180,3)

def getHighFreqMask(img):
    highfreqimg = defaulthighp(img)
    #cv2.imshow('highfreqimg',highfreqimg)
    cannyimg = cv2.Canny(highfreqimg,100,180,3)
    cv2.imshow('cannyimg',cannyimg)
    enforced = cv2.filter2D(cannyimg, -1, np.ones((40,40))/800)
    cv2.imshow('enforced0',enforced)
    enforced = cv2.filter2D(enforced, -1, np.ones((20,20))/350)
    cv2.imshow('enforced1',enforced)
    """enforced = cv2.filter2D(enforced, -1, np.ones((5,5))/16)
    cv2.imshow('enforced2',enforced)
    enforced = cv2.filter2D(enforced, -1, np.ones((3,3))/6)
    cv2.imshow('enforced3',enforced)"""

    ret,threshenforced = cv2.threshold(enforced,100,255,cv2.THRESH_BINARY)
    #threshenforced = cv2.erode(threshenforced,np.ones((3,3)),iterations = 1)
    #threshenforced = cv2.dilate(threshenforced,np.ones((3,3)),iterations = 3)
    #threshenforced = cv2.erode(threshenforced,np.ones((3,3)),iterations = 3)
    cv2.imshow('threshenforced',threshenforced)

    orgcanny = getEdges(img)
    #orgcanny = cv2.Canny(img,70,200,3)
    threshenforced = 255 - threshenforced
    reducedcanny = cv2.bitwise_and(orgcanny,orgcanny,mask = threshenforced)
    cv2.imshow('reduced',reducedcanny)

    kernel3 = np.array([[0, 1, 0],
                        [1, 4, 1],
                        [0, 1, 0]],np.uint8)
    kernel5 = np.array([[0,  0,  1,  0,  0],
                        [0,  2,  4,  2,  0],
                        [1,  4,  8,  4,  1],
                        [0,  2,  4,  2,  0],
                        [0,  0,  1,  0,  0]],np.uint8)

    reducedcanny.astype(np.float32)
    #reducedcanny = cv2.dilate(reducedcanny,kernelx,iterations = 1)
    reducedcanny = cv2.filter2D(reducedcanny, -1, kernel5/13)
    reducedcanny = cv2.filter2D(reducedcanny, -1, kernel3/4)
    
    detailCanny = getEdges2(img)
    detailCanny.astype(np.float32)
    #detailCanny = cv2.filter2D(detailCanny, -1, kernel3/4)
    detailCanny = cv2.filter2D(detailCanny, -1, kernel3/4)

    reducedcanny = cv2.bitwise_or(reducedcanny, detailCanny)

    cv2.imshow('reduced2',reducedcanny)
    newimg = img.copy()
    
    reducedcanny = 1-reducedcanny/255
    newimg[:,:,0] = newimg[:,:,0] * reducedcanny
    newimg[:,:,1] = newimg[:,:,1] * reducedcanny
    newimg[:,:,2] = newimg[:,:,2] * reducedcanny
    #newimg[reducedcanny == 255] = 0

    cv2.imshow('orgcanny',newimg)

    return threshenforced



#def dirac(img):
#    shape = img.shape
#    return applybandpassfilter(img,40*shape[0],40*shape[1],60*shape[0],60*shape[0])


orgimg = cv2.imread('C:/Users/Max/Pictures/vlcsnap-error127.png')
#orgimg = cv2.imread('C:\\Users\\Max\\Desktop\\Costa Rica\\Phone\\20190204_100651.jpg')


shape = orgimg.shape

cv2.namedWindow('image')
haschanged = True
def change(x):
    global haschanged
    #print("change called")
    haschanged = True
"""cv2.createTrackbar('highpy','image',0,shape[0],change)
cv2.createTrackbar('highpx','image',0,shape[1],change)
cv2.createTrackbar('lowpy','image',0,shape[0],change)
cv2.createTrackbar('lowpx','image',0,shape[1],change)"""
cv2.createTrackbar('highp','image',0,100,change)
cv2.createTrackbar('lowp','image',0,100,change)

filteredImg = None
while(True):
    #print("Run loop")
    #print(haschanged)
    if(haschanged):
        #print("apply")
        """highpy = cv2.getTrackbarPos('highpy','image')
        highpx = cv2.getTrackbarPos('highpx','image')
        lowpy = cv2.getTrackbarPos('lowpy','image')
        lowpx = cv2.getTrackbarPos('lowpx','image')"""
        highp = cv2.getTrackbarPos('highp','image')
        lowp = cv2.getTrackbarPos('lowp','image')

        filteredImg = applybandpassfilter(orgimg,highp*shape[0]/200,highp*shape[1]/200,lowp*shape[0]/200,lowp*shape[1]/200)
        #filteredImg = applybandpassfilter(orgimg,highpy/2,highpx/2,lowpy/2,lowpx/2)
        haschanged = False
        if(orgimg is not None):
            cv2.imshow('orgimg',orgimg)
        if(filteredImg is not None):
            #cv2.imshow('highfreq',getHighFreqMask(orgimg))
            cv2.imshow('highfreq',cv2.resize(trippyfilter2(orgimg,lowp*2.55),(int(shape[1]/2),int(shape[0]/2))))
            
            """cv2.imshow('filteredImg',filteredImg)
            enforced = cv2.filter2D(filteredImg, -1, np.ones((10,10))/50)
            enforced = cv2.filter2D(enforced, -1, np.ones((10,10))/30)
            enforced = cv2.filter2D(enforced, -1, np.ones((10,10))/50)
            #enforced = cv2.filter2D(enforced, -1, np.ones((10,10))/50)
            #enforced = cv2.filter2D(enforced, -1, np.ones((10,10))/50)

            cv2.imshow('enforced',enforced)

            grayenforced = cv2.cvtColor(enforced,cv2.COLOR_BGR2GRAY)
            ret,threshenforced = cv2.threshold(grayenforced,150,255,cv2.THRESH_BINARY)
            threshenforced = cv2.erode(threshenforced,np.ones((3,3)),iterations = 1)
            threshenforced = cv2.dilate(threshenforced,np.ones((3,3)),iterations = 3)
            threshenforced = cv2.erode(threshenforced,np.ones((3,3)),iterations = 3)
            cv2.imshow('threshenforced',threshenforced)

            sobel = applysobelfilter(orgimg)

            cv2.imshow('sobel',sobel)

            canny = cv2.Canny(filteredImg,100,180,3)
            cv2.imshow('canny',canny)

            #threshenforced = 255 - threshenforced

            #reducedcanny = cv2.bitwise_and(canny,canny,mask = threshenforced)
            #cv2.imshow('reduced',reducedcanny)

            #newimg = orgimg.copy()
            #newimg[reducedcanny == 255] = 0

            #cv2.imshow('orgcanny',newimg)

            #enforced = cv2.filter2D(enforced, -1, np.ones((5,5))/4)
            #enforced = cv2.filter2D(enforced, -1, np.ones((5,5))/4)"""
            
        #channels = cv2.split(filteredImg)
        #for chann in range(0,3):
        #    cv2.imshow('filteredImg:' + str(chann),filteredImg[:,:,chann])

    k = cv2.waitKey(100)
    if(k > 0):
        #print("Break loop")
        break

cv2.destroyAllWindows()