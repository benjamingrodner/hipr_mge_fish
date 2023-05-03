import matplotlib
matplotlib.use('Agg')
import pylab as pl
import argparse
import openslide
import json
import skimage.io
import os
import os.path
import re
import tempfile
import Histology as his
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i','--input', help = 'Input description')
ap.add_argument('-f','--file', help = 'Image file')
ap.add_argument('-o','--out', help = 'Output Image file')
ap.add_argument('-n','--name', help = 'file name')
ap.add_argument('-s','--size', action='store_true', help = 'file name')
args = vars(ap.parse_args())

input = ""
file = ""

if (args['input']):
    input = args['input'].strip()
if (args['file']):
    file = args['file']

def getFiles(input):
    res = []
    index = 0
    with open("colon-images.txt", "r") as f:
        for line in f:
            line = line.strip();
            ll1 = line.split("/");
            if len(input) > 11 and ll1[-1].startswith(input):
                res.append(line)
            if len(input) > 0 and (not input.startswith("TCGA")) and index == int(input):
                res.append(line)
            index += 1
    return res

def parseInput(input):
    if input.startswith("http"):
    	img = skimage.io.imread((input))[:, :, :3]
	return img
    l1 = input.split(",")
    name = l1[0].strip();
    list = getFiles(name)
    svsfile = None
    if len(list) > 0:
        svsfile = list[0]
    w,h,x,y = 1000,1000,500,500
    coord = str(w) + "x" + str(h) + "+" +  str(x) + "+" + str(y)
    if len(l1) > 1:
        coord = l1[1].strip()
    l1 = re.split("[x+\s]", coord)
    if len(l1) == 1:
        w = h = int(l1[0])
    if len(l1) == 2:
        w,h = int(l1[0]),int(l1[1])
    if len(l1) == 4:
        w,h,x,y = [int(l1[i]) for i in range(4)]
    if (svsfile != None):
        slide = openslide.OpenSlide(svsfile)
        img = slide.read_region((x,y),0,(w,h))
	return his.PIL2array(img)
    return None

if (args['out'] and args['name']):
    if (args['name'].startswith("http") or
            args['name'].find(",") >= 0):
        img = parseInput(args['name'])
        if (img is not None):
            skimage.io.imsave(args['out'], img)
            exit(0)
    list = getFiles(args['name'])
    if len(list) > 0:
        svsfile = list[0]
        slide = openslide.OpenSlide(svsfile)
        img = slide.associated_images['thumbnail']
        width, height = img.size
        w, h = slide.dimensions
	img.save(args['out'], "png")
    exit(0)

if (args['size'] and args['name']):
    list = getFiles(args['name'])
    if len(list) > 0:
        svsfile = list[0]
        slide = openslide.OpenSlide(svsfile)
        img = slide.associated_images['thumbnail']
        width, height = img.size
        w, h = slide.dimensions
        print "Size:", width, height, w, h 
    exit(0)

#print args
imInput = None

if file == "":
   imInput = parseInput(input)

if file != "" and os.path.isfile(file) and os.access(file, os.R_OK):
    imInput = skimage.io.imread(file)[:, :, :3]

if imInput is None:
    exit(1)

imgName = input.split(",")[0]
print "Size of tile: ", imInput.shape
genelist = ["MUC2", "KRT20", "CDX2"]

print "<h2>Original Image</h2>"
res = his.array2base64(imInput)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

# perform reinhard color normalization
imNmzd = his.reinhard_array(imInput)

print "<h2>Normalized Image</h2>"
res = his.array2base64(imNmzd)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>Smoothened Image</h2>"
smoothImg = his.smoothingImg(imNmzd)
res = his.array2base64(smoothImg)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>BlueRatio Image</h2>"
BRimg = his.getBlueRatio(smoothImg)
res = his.mat2base64(BRimg)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>OTSU Threshold Greyscale Image</h2>"
otsugrey, otsu, otsuThr = his.otsuThreshGrey(BRimg, 200)
res = his.mat2base64(otsugrey)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>OTSU Threshold Image</h2>"
res = his.mat2base64(otsu)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

## create stain to color map
#stainColorMap = {
        #'hematoxylin': [0.65, 0.70, 0.29],
        #'eosin':       [0.07, 0.99, 0.11],
        #'dab':         [0.27, 0.57, 0.78],
        #'null':        [0.0, 0.0, 0.0]
        #}

## specify stains of input image
#stain_1 = 'hematoxylin'   # nuclei stain
#stain_2 = 'eosin'         # cytoplasm stain
#stain_3 = 'null'          # set to null of input contains only two stains

## create stain matrix
#W = np.array([stainColorMap[stain_1], 
    #stainColorMap[stain_2], 
    #stainColorMap[stain_3]]).T

## perform standard color deconvolution
#imDeconvolved = his.color_deconvolution(imNmzd, W)

## get nuclei/hematoxylin channel
#imNucleiStain = imDeconvolved.Stains[:, :, 0].astype(np.float)

#H = imDeconvolved.Stains[:, :, 0]
#E = imDeconvolved.Stains[:, :, 1]

#print "<h2>Hematoxylin Image</h2>"
#res = his.mat2base64(H)
#print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

#print "<h2>Eosin Image</h2>"
#res = his.mat2base64(E)
#print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

##print "<h2>Detected edges</h2>"
##imgE = his.detectEdges(H)
##res = his.mat2base64(imgE)
##print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

#print "<h2>Threshold 1 Image</h2>"
#imgSM = his.imageThreshold(H)
#res = his.mat2base64(imgSM)
#print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

##print "<h2>Overlay Image</h2>"
##markers = his.imageWatershed(imgSM)
##imgFL = his.overlayNuclei(markers, imNmzd)
##res = his.mat2base64(imgFL)
##print "<img width=\"1000px\" src=\"data:image/png;base64," + res + "\"/><br/>"

#print "<h2>Filter 1 Image</h2>"
#imgFL = his.imageFilter(imgSM)
#res = his.mat2base64(imgFL)
#print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

#print "<h2>Binary Image</h2>"
#binary = his.grayscale2binary(imgFL, method = "otsu")
#res = his.mat2base64(binary)
#print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>Watershed Image</h2>"
watershed = his.watershedAlgo(BRimg, otsu, np.zeros_like(otsu))
res = his.mat2base64(watershed)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>Expanded Marker Image</h2>"
otsuIncreased, otsuThrDec = his.expandOtsuGlands(BRimg, otsuThr, otsu, watershed)
otsuIncreased = his.removeSmallHoles(otsuIncreased)
otsuIncreased = his.removeSmallObjects(otsuIncreased)
otsuIncreased = his.binaryClosing(otsuIncreased)
otsuIncreased = his.binaryOpening(otsuIncreased)
res = his.mat2base64(otsuIncreased)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>OTSU Threshold Corrected Image</h2>"
otsuDecresed = his.thresholdOtsuGlands(BRimg, otsuThrDec, otsuIncreased)
otsuDecresed = his.removeSmallHoles(otsuDecresed)
otsuDecresed = his.removeSmallObjects(otsuDecresed)
otsuDecresed = his.binaryClosing(otsuDecresed)
otsuDecresed = his.binaryOpening(otsuDecresed)
res = his.mat2base64(otsuDecresed)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>Watershed Image</h2>"
watershed = his.watershedAlgo(BRimg, otsuDecresed, otsuIncreased)
watershed = his.removeSmallHoles(watershed)
watershed = his.removeSmallObjects(watershed)
watershed = his.binaryClosing(watershed)
watershed = his.binaryOpening(watershed)
res = his.mat2base64(watershed)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

#histogram, glands = his.segment_gland_region(watershed)
#res = his.mat2base64(histogram)
#print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"
#res = his.mat2base64(glands)
#print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"
  
print "<h2>Rectangle Marked Image</h2>"
imgMarked, Props1, Props2 = his.markGlands(watershed, imInput)
res = his.mat2base64(imgMarked)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"
  
print "Your Image: ", imgName, "<br />"
print "No of Circular Nuclie (Red): ", len(Props1), "<br />"
print "No of Elongated Nuclie (Yellow): ", len(Props2), "<br />"
his.getGeneExpression(imgName, genelist)
his.printFeatures(watershed)

print "<h2>Contours Image</h2>"
contourImg = his.findContours(watershed, BRimg)
res = his.mat2base64(contourImg)
print "<img width=\"500px\" src=\"data:image/png;base64," + res + "\"/><br/>"

print "<h2>Overlay Image</h2>"
labelledImg = his.labelImage(watershed)
imgFL = his.overlayNuclei(labelledImg, imNmzd)
res = his.mat2base64(imgFL)
print "<img width=\"1000px\" src=\"data:image/png;base64," + res + "\"/><br/>"

exit(0)

#print "<h2>Segmented Image</h2>"
#nSeg = his.segmentNuclei(imgSM)
#imgFL = his.overlayNuclei(nSeg, imInput)
#res = his.mat2base64(imgFL)
#print "<img width=\"1000px\" src=\"data:image/png;base64," + res + "\"/><br/>"


