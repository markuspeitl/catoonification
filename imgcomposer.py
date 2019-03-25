import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dstcomposer',help='name of the filter to be executed')
parser.add_argument('srcimages',nargs='+',help='path of the image to be filterd, requires images to be non normalized and in range 0 - 255')
parser.add_argument('dstimage',help='path of the image destination')

args = parser.parse_args()

print("imgcomposer started called")
dstcomposer = None
srcimages = None
dstimage = None

if(args.dstcomposer != None):
    dstcomposer = args.dstcomposer

if(args.srcimages != None):
    srcimages = args.srcimages

if(args.dstimage != None):
    dstimage = args.dstimage

print(srcimages)

if srcimages is not None:
    srcimgbuffers = []
    for imagePath in srcimages:
        srcimagebuf = cv2.imread(imagePath)
        srcimgbuffers.append(srcimagebuf)

    dstimagebuf = srcimgbuffers[0].copy()
    dstimagebuf = dstimagebuf/255
    dstimagebuf.astype(np.float32)
    
    for i in range(1,len(srcimgbuffers)):
        normimg = srcimgbuffers[i]/255
        dstimagebuf = dstimagebuf * normimg

    dstimagebuf = dstimagebuf*255

    #for imagePath in srcimages:
    #    os.remove(imagePath)

    if(dstimage):
        print("writing image to path: " + dstimage)
        cv2.imwrite(dstimage,dstimagebuf)




