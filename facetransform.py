import cv2
import numpy as np
import argparse
import os

#orgimg = cv2.imread('C:/Users/Max/Pictures/1909374_727114680641985_853457871_o.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/53146095_10155809091156428_4847299437430571008_n.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/Image.png')
#orgimg = cv2.imread('C:/Users/Max/Pictures/20190212_121745.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/14448806_1216594781693970_8494145683709789122_n.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/7773_1046764575343659_1669231472579208439_n.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/11828579_979384208748363_3243964694848316649_n.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/1004623_594852887201499_861679947_n.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/P1060245.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/20190205_155101.jpg')
#orgimg = cv2.imread('C:/Users/Max/Pictures/20190205_151042.jpg')
orgimg = cv2.imread('C:/Users/Max/Pictures/20190204_124204.jpg')

width,height,ch = orgimg.shape
img = orgimg

print(orgimg.shape)

#face_cascade = cv2.CascadeClassifier('../../data/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('../../data/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('C:/Users/Max/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Max/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_eye.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(face_cascade)

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

def meanquadrants(img):
    
    outh = 2
    outw = 2
    h,w = img.shape

    output = np.zeros((outh,outw),np.float32)

    winh = int(h/outh)
    winw = int(w/outw)

    for y in range(1,outh+1):
        for x in range(1,outw+1):
            output[y-1,x-1] = np.mean(img[(y-1)*winh:y*winh,(x-1)*winw:x*winw])

            #cv2.imshow('circularweight' + str(np.random.random()),cv2.resize(img[(y-1)*winh:y*winh,(x-1)*winw:x*winw],(300,300)))
            #print(img[(y-1)*winh:y*winh,(x-1)*winw:x*winw])

    #print(output)
    centerwindow = img[winh-int(winh/2):winh+int(winh/2),winw-int(winw/2):winw+int(winw/2)]
    #print("centermean:" + str(np.mean(centerwindow)))
    #cv2.imshow('centermean' + str(np.random.random()),cv2.resize(centerwindow,(300,300)))
    

    dia1corr = output[0,0]/output[1,1]
    if(dia1corr > 1):
        dia1corr = 2 - dia1corr
    dia2corr = output[1,0]/output[0,1]
    if(dia2corr > 1):
        dia2corr = 2 - dia2corr

    #normoutput = output/maxval
    #np.prod(normoutput)
    return (dia1corr + dia2corr)/2
    #return output

# Slide Mask with certain shape over the image and check for best match
# Repeat for different kernel sizes and find best match (highest template alignment)
# used to find the position of template
# Perfect match should be 1, so the closer the match the closer to 1 
def cascadeCicularMask(img,showind):

    print("cascading for "+ str(showind))

    orgimg = img
    img = 255 - img
    #img[img < 200] = 0
    h,w = img.shape
    print(img.shape)
    minside = min(w,h)
    print(minside)

    #fulllayers = 10
    #effectivelayers = 1

    steps = 20

    
    resultSizes = []
    resultGradients = []
    symmetryVals = []
    symmetryCirGrad = []
    symmetryGrad = []
    valueSums = []

    startksize = int(minside/2)
    #startksize = int(minside)
    minksize = 5
    stepsize = (startksize-minksize)/(steps-1)
    if stepsize < 1:
        stepsize = 1
    
    cntksize = startksize

    effsteps = int((startksize - minksize)/stepsize) + 1
    resultLayers = np.zeros((h,w,effsteps),np.uint8)

    i = 0
    while cntksize >= minksize:
    #for i in range(0,steps):
        #ksize = startksize - int(stepsize*i)

        ksize = int(cntksize)
        print("selk: " + str(ksize))

        if(ksize % 2 is 0):
            ksize += 1

        circlemask = createCirclularMask(ksize,cthickness=20000)
        #Slide mask
        #print(circlemask)

        #cv2.imshow('circlemask' + str(i),cv2.resize(circlemask*ksize*ksize*255,(300,300)))
        circularweight = cv2.filter2D(img, -1, circlemask)

        bestfindinlayer = np.unravel_index(np.argmax(circularweight, axis=None), circularweight.shape)
        #print("best index "+ str(i) + ":" + str(bestfindinlayer))
        sh = bestfindinlayer[0] - int(ksize/2)
        eh = bestfindinlayer[0] + int(ksize/2 + 0.5)
        sw = bestfindinlayer[1] - int(ksize/2)
        ew = bestfindinlayer[1] + int(ksize/2 + 0.5)
        if sh >= 0 and eh < h and sw >= 0 and ew < w:
            print(str(sh) + " - " + str(eh) + " - " + str(sw) + " - " + str(ew))
            maskedroi = img[sh:eh,sw:ew]
            circlemaskfull = createCirclularMask(ksize,cthickness=20000)*ksize*ksize*255
            circlemaskfull = circlemaskfull.astype(np.uint8)
            maskedroi = maskedroi.astype(np.uint8)
            gradient = sobelFilter(maskedroi,ksize=3)
            #print(ksize)
            #print(maskedroi.shape)
            #print(circlemaskfull.shape)

            circulargradient = circleSobel(maskedroi)
            #cv2.imshow('circulargradient' + str(showind) + "s:" + str(i),cv2.resize(circulargradient,(300,300)))
            #invcirclemaskfull = 255 - circlemaskfull
            #maskedgrad = cv2.bitwise_and(gradient,gradient,mask=circlemaskfull)
            maskedgrad = cv2.bitwise_and(gradient,circlemaskfull)
            maskedrr = cv2.bitwise_and(maskedroi,circlemaskfull)
            valsum = np.sum(maskedrr)
            gradsum = np.sum(maskedgrad)
            circlegradsum = np.sum(circulargradient)

            print("ksize res "+ str(ksize) + " step: " + str(i))

            #
            """if(showind == 0):
                print("ksize res "+ str(ksize) + " step: " + str(i))
                #print("sum gradsum:")
                #print(meanquadrants(maskedgrad))
                print("sum valsum:")
                #print(meanquadrants(maskedroi))
                print(meanquadrants(maskedrr))
                print("sum circulargradient:")
                print(meanquadrants(circulargradient))
                print("----------------------------------")
            """
            symmetryGrad.append(meanquadrants(maskedgrad))
            symmetryVals.append(meanquadrants(maskedrr))
            symmetryCirGrad.append(meanquadrants(circulargradient))
            valueSums.append(valsum)
            
            #cv2.imshow('gradients' + str(i),cv2.resize(gradient,(300,300)))
            #cv2.imshow('circlemaskfull' + str(i),cv2.resize(circlemaskfull,(300,300)))
            #print("gradsum: " + str(gradsum/(ksize*ksize)))
            #print("valsum: " + str(valsum/(ksize*ksize)))
            #print("circlegradsum: " + str(circlegradsum/(ksize*ksize)))
            #cv2.imshow('maskedroi' + str(i),cv2.resize(maskedroi,(300,300)))
            #cv2.imshow('gradient' + str(i),cv2.resize(maskedgrad,(300,300)))

            #cv2.imshow('circularweight' + str(i),cv2.resize(circularweight,(300,300)))
            #print(circularweight)
            resultLayers[:,:,i] = circularweight
            #resultLayers.append(circularweight)
            resultSizes.append(ksize)

            i += 1

        cntksize -= stepsize

    bestfind = np.unravel_index(np.argmax(resultLayers, axis=None), resultLayers.shape)
    #bestfind = np.argmax(resultLayers, axis=None)
    print(bestfind)
    print(resultLayers[bestfind[0],bestfind[1],bestfind[2]])
    print("-------------------")
    #print(resultLayers[bestfind[1],bestfind[0],bestfind[2]])

    pointsimg = orgimg.copy()
    #print("resultLayers.shape[2]: " + str(resultLayers.shape[2]))
    for i in range(0,len(resultSizes)):
        bestinlayer = np.argmax(resultLayers[:,:,i], axis=None)
        bestfindinlayer = np.unravel_index(np.argmax(resultLayers[:,:,i], axis=None), resultLayers[:,:,i].shape)

        #print(bestfindinlayer)
        #print(resultLayers[bestfindinlayer[0],bestfindinlayer[1],i])
        layerk = resultSizes[i]

        drawimg = np.zeros(resultLayers[:,:,i].shape,np.uint8)
        cv2.circle(drawimg,(bestfindinlayer[1],bestfindinlayer[0]),int(layerk/2),255,thickness=1)
        #cv2.circle(drawimg,(bestfindinlayer[0],bestfindinlayer[1]),int(layerk/2),255,thickness=1)
        #resultLayers[:,:,i] = cv2.bitwise_or(resultLayers[:,:,i],drawimg)
        resultLayers[:,:,i] = cv2.bitwise_or(orgimg,drawimg)
        resultLayers[bestfindinlayer[0],bestfindinlayer[1],i] = 255

        #if i == 5:

        pointsimg[bestfindinlayer[0],bestfindinlayer[1]] = 255

        #cv2.imshow('circularweight' + str(showind) + " s:"  + str(i),cv2.resize(resultLayers[:,:,i],(300,300)))
        cv2.imshow('circularweight' + str(showind) + " s:"  + str(i),resultLayers[:,:,i])

        #print(bestinlayer)
        
        #print(resultLayers[bestfindinlayer[1],bestfindinlayer[0],i])

    #orgimg[bestfind[0],bestfind[1]] = 255
    #cv2.imshow('withpoint' + str(showind) + " s:" + str(i),cv2.resize(orgimg,(300,300)))

    maxValSymm = max(symmetryVals)
    bestValSymm = symmetryVals.index(maxValSymm)
    print("Best value symmetry: " + str(bestValSymm))
    print("Best circular gradient symmetry: " + str(symmetryCirGrad.index(max(symmetryCirGrad))))
    print("Best value sum: " + str(valueSums.index(max(valueSums))))
    print("Lowest gradient symmetry: " + str(symmetryGrad.index(min(symmetryGrad))))

    pointsimg[bestfind[0],bestfind[1]] = 255
    cv2.imshow('withpoints' + str(showind) + " s:" + str(i),cv2.resize(pointsimg,(300,300)))

    """bestk = resultSizes[bestfind[2]]

    orgimg[bestfind[0],bestfind[1]] = 255

    cv2.circle(orgimg,(bestfind[0]-int(bestk/2),bestfind[1]+int(bestk/2)),int(bestk/2),255,thickness=-1)
    #cv2.circle(orgimg,(bestfind[1]-int(bestk/2),bestfind[0]+int(bestk/2)),int(bestk/2),255,thickness=-1)
    cv2.imshow('withpoint' + str(i),cv2.resize(orgimg,(300,300)))"""
    

def sobelFilter(img,ksize=3):
    #print("sobelFilter")
    img = img.astype(np.float32)
    grad_x = cv2.Sobel(img,cv2.CV_16U,1,0,ksize=ksize)
    grad_y = cv2.Sobel(img,cv2.CV_16U,0,1,ksize=ksize)
    dstimg = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5,0)
    scaled = cv2.convertScaleAbs(dstimg)
    scaled = scaled.astype(np.uint8)
    #cv2.imshow('dstimg',scaled)
    return scaled

def cascade(img,ind):
    w,h = img.shape
    resizeto = 5
    xfactor = int(w/resizeto)
    yfactor = int(h/resizeto)
    minieye = cv2.resize(img,(resizeto,resizeto),0)
    index = np.unravel_index(np.argmin(minieye, axis=None), minieye.shape)
    print(index)
    ws = index[0] * xfactor
    we = ws + xfactor
    hs = index[1] * yfactor
    he = hs + yfactor

    smaller = img[hs:he,ws:we]

    cv2.imshow('smaller' + str(ind),cv2.resize(smaller,(300,300)))

    #minieye[index[0],index[1]] = 255

def circleSobel(img):
    kernelx = np.array([[ 1, 0, -1],
                        [ 2, 0, -2],
                        [ 1, 0, -1]])

    kernely = np.array([[ 1, 2, 1],
                        [ 0, 0, 0],
                        [-1,-2,-1]])

    kernelz = np.array([[ 2, 1, 0],
                        [ 1, 0,-1],
                        [ 0,-1,-2]])

    kernelm = np.array([[-2,-1, 0],
                        [-1, 0, 1],
                        [ 0, 1, 2]])
    img = img.astype(np.float32)
    sobelx = cv2.filter2D(img, -1, kernelx)
    sobely = cv2.filter2D(img, -1, kernely)
    sobelz = cv2.filter2D(img, -1, kernelz)
    sobelm = cv2.filter2D(img, -1, kernelm)
    dstimg = sobelx * 0.25 + sobely * 0.25 + sobelz * 0.25 + sobelm * 0.25
    #dstimg = cv2.addWeighted(sobelx, 0.25, sobely, 0.25, sobelz, 0.25, sobelm, 0.25,0)
    #fullsobel = cv2.bitwise_or(sobelx, sobely, sobelz, sobelm)
    fullsobel = cv2.convertScaleAbs(dstimg)
    fullsobel = fullsobel.astype(np.uint8)

    return fullsobel

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_face = img[y:y+h, x:x+w]
    face_size = w*h
    eyes = eye_cascade.detectMultiScale(roi_gray)
    #print(eyes)

    grayface = cv2.cvtColor(roi_face[30:roi_face.shape[0]-30,30:], cv2.COLOR_BGR2GRAY)
    gh,gw = grayface.shape
    grayface = grayface[100:grayface.shape[0]-100,200:grayface.shape[1]-200]

    grayfacel = grayface[:,0:int(grayface.shape[1]/2)]
    grayfacel = cv2.resize(grayfacel,(int(gh/4),int(gw/2)))

    grayfacer = grayface[:,int(grayface.shape[1]/2):]
    grayfacer = cv2.resize(grayfacer,(int(gh/4),int(gw/2)))

    cascadeCicularMask(grayfacel,0)
    cascadeCicularMask(grayfacer,1)

    cv2.imshow('face',roi_face)
    cnt = 0

    for (ex,ey,ew,eh) in eyes:

        if(ew * eh > face_size/1000):

            #cv2.rectangle(roi_face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_eye = roi_face[ey:ey+eh,ex:ex+ew]
            #cv2.imshow('eye' + str(cnt),roi_eye)

            grayeye = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY)
            blurredgrayeye = cv2.GaussianBlur(grayeye, (3, 3),0.5)
            #eyeedges = cv2.Canny(blurredgrayeye,60,255,3)
            eyeedges = sobelFilter(grayeye)
            #eyeedges0 = circleSobel(roi_eye[:,:,0])
            #eyeedges1 = circleSobel(roi_eye[:,:,1])
            #eyeedges2 = circleSobel(roi_eye[:,:,2])
            #eyeedges = cv2.bitwise_or(eyeedges0,eyeedges1,eyeedges2)

            #cascadeCicularMask(grayeye,cnt)

            #cascade(grayeye,cnt)

            """minieye = cv2.resize(grayeye,(5,5),0)
            index = np.unravel_index(np.argmin(minieye, axis=None), minieye.shape)
            print(index)
            minieye[index[0],index[1]] = 255

            cv2.imshow('minieye' + str(cnt),cv2.resize(minieye,(300,300)))"""

            eyeedges[eyeedges > 50] = 255
            eyeedges[eyeedges <= 50] = 0
            #cv2.imshow('eyeedges' + str(cnt),cv2.resize(eyeedges,(300,300)))

            #print(grayeye.mean())

            baseIntensity = grayeye.mean()

            mask0 = cv2.inRange(grayeye, 0, 0)
            mask1 = cv2.inRange(grayeye, 0, baseIntensity * 0.6)
            mask = cv2.bitwise_or(mask0,mask1)
            #cv2.imshow('eyemask' + str(cnt),cv2.resize(mask,(300,300)))

            #lower_white = np.array([0,140,0])
            #upper_white = np.array([180,255,120])

            #bgclose = roi_eye[:,:,0]/roi_eye[:,:,1]
            #grclose = roi_eye[:,:,1]/roi_eye[:,:,2]
            #brclose = roi_eye[:,:,0]/roi_eye[:,:,2]

            """fullcls = ((bgclose + grclose + brclose)/3)
            newimg = np.zeros(roi_eye.shape,np.uint8)
            newimg0 = np.zeros(roi_eye.shape,np.uint8)
            newimg1 = np.zeros(roi_eye.shape,np.uint8)
            newimg0[0.9 < fullcls] = 255
            newimg1[1.1 > fullcls] = 255
            newimg = cv2.bitwise_and(newimg0,newimg1)"""

            #hsveye = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2HLS)
            #mask = cv2.inRange(hsveye, lower_white, upper_white)
            #cv2.imshow('eyegray' + str(cnt),hsveye)
            #mask = cv2.inRange(hsveye, 120, 255)
            #cv2.imshow('eyemask' + str(cnt),mask)
            #cv2.imshow('eyemask' + str(cnt),newimg)

            cnt += 1


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()