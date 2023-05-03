import matplotlib
matplotlib.use('Agg')
import pylab as pl
import cv2
import openslide
import StepMiner as sm
import StepMinerNew as smn
import re
import numpy as np
import scipy as sp
import scipy.ndimage
import pandas as pd
import Image
import base64
import io
import colorsys
import random
import skimage.io
import skimage.measure
import skimage.color
import max_clustering as mc
import nucleus
    
def histogram(arr):
    fig = pl.figure(figsize=(6,6))
    pl.hist(arr)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0);
    img = Image.open(buf)
    return PIL2array(img)

def plot(x, y):
    fig = pl.figure(figsize=(6,6))
    pl.plot(x, y)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0);
    img = Image.open(buf)
    return PIL2array(img)

def segment_gland_region(img):
    labelled_img, nbr_objects = nucleus.label_image(img)
    props = propsFromImg(img)
    centroids = nucleus.get_centroid_list(props)
    mini_arr = nucleus.get_minimum_dist_for_nuclie(centroids)
    dist_Thr = (sum(mini_arr)*1.)/len(mini_arr)
    #print dist_Thr
    for i in range(len(centroids)):
	if mini_arr[i] >= dist_Thr:
	    img[labelled_img == props[i]["label"]] = 255
    return histogram(mini_arr), img
    
def imageFilter(img):
    im_mask = sp.ndimage.morphology.binary_fill_holes(img < 250)
    im_dmap = sp.ndimage.morphology.distance_transform_edt(im_mask)
    im_dmap[im_dmap > 3] = 3
    im_dmap[im_dmap < 2] = 0
    I1 = im_dmap * (255 - img)
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255).astype(np.uint8)
    I2 = 255 - I2
    return I2
  
def otsuThresh(img, radius):
    from skimage.filters import threshold_otsu, rank
    selem = skimage.morphology.disk(radius)
    local_otsu = rank.otsu(img, selem)
    I1 = (img > local_otsu).astype(int)
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    return I2
    
def otsuThreshGrey(img, radius):
    img = 255 - img
    from skimage.filters import threshold_otsu, rank
    selem = skimage.morphology.disk(radius)
    local_otsu = rank.otsu(img, selem)
    I1 = (img > local_otsu).astype(int)
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    img[img < local_otsu] = 0
    return 255-img, 255-I2, local_otsu

def imgThr2(img):
    T1 = otsuThresh(255-img, 200)
    T2 = otsuThresh(255-img, 15)
    I1 = np.sqrt(np.multiply(T1, T2))
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    return (255-I2)
  
def globalOTSU(img):
    from skimage.filters import threshold_otsu
    img = 255 - img
    thresh = threshold_otsu(img)
    I1 = (img > thresh).astype(int)
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    return 255 - I2

def thresholdOtsuGlands(BRimg, otsuThr, binary):
    blobs = []
    props = propsFromImg(binary)
    totalArea = np.sum([x["area"] for x in props])
    areaTh = (totalArea/len(props))*2
    props, big_props = removeUnsegmentedBlobs(props, areaTh)
    blobs = nucleus.get_bounding_box_list(big_props, (1,1))
    otsuThrInc = otsuThr.copy()
    for blob in blobs:
	img_section = big_props[blobs.index(blob)]["image"]
	BRimgtile = BRimg[blob[0]:blob[2],blob[1]:blob[3]]
	otsuThrtile = otsuThrInc[blob[0]:blob[2],blob[1]:blob[3]]
	diff = np.subtract(255-BRimgtile , otsuThrtile)
	diff = diff[img_section == np.max(img_section)]
	mean = np.mean(diff)
	std = np.std(diff)
	#print "Mean: ", mean
	#print "Std Dev.: ", std	
	sec_max = np.max(img_section)
	for i in range(blob[3] - blob[1]):
	    for j in range(blob[2] - blob[0]):
		if img_section[j][i] == sec_max:
		    otsuThrInc[blob[0]+j,blob[1]+i] += int(mean + std)
    I1 = ((255 - BRimg) > otsuThrInc).astype(int)
    I2 = 255 - (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    return I2
  
def expandOtsuGlands(BRimg, otsuThr, binary, region):
    blobs = []
    props = propsFromImg(binary)
    blobs = nucleus.get_bounding_box_list(props, (1,1))
    labelled_regions, nbr_objects = nucleus.label_image(region)
    region_props = nucleus.get_label_props(labelled_regions)
    otsuThrDec = otsuThr.copy()
    for blob in blobs:
	img_section = props[blobs.index(blob)]["image"]
	label = props[blobs.index(blob)]["label"]
	BRimgtile = BRimg[blob[0]:blob[2],blob[1]:blob[3]]
	otsuThrtile = otsuThrDec[blob[0]:blob[2],blob[1]:blob[3]]
	diff = np.subtract(255-BRimgtile , otsuThrtile)
	diff = diff[img_section == np.max(img_section)]
	mean = np.mean(diff)
	std = np.std(diff)
	otsuThrDec[labelled_regions == label] -= int(std)
    I1 = ((255 - BRimg) > otsuThrDec).astype(int)
    I2 = 255 - (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    return I2, otsuThrDec
  
def watershedAlgo(img, sure_fg, sure_bg):
    sure_fg = 255 - sure_fg
    sure_bg = 255 - sure_bg
    #sure_bg[sure_bg < 100] = 0
    #sure_bg[sure_bg >= 100] = 255
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = connectedComponents(sure_fg)
    markers = markers.astype(np.int32)
    markers = markers+1
    markers[unknown==255] = 0
    from skimage.morphology import watershed
    output = watershed(img, markers, watershed_line = True)
    sure_bg[output == 0] = 0
    return 255 - sure_bg
  
def smoothingImg(img):
    smooth = cv2.blur(img, (3,3))
    return smooth
  
def binaryOpening(img, mesh_size = np.ones((1,1))):
    I1 = skimage.morphology.binary_opening(img, mesh_size)
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    return I2

def binaryClosing(img, mesh_size = np.ones((1,1))):
    I1 = skimage.morphology.binary_closing(img, mesh_size)
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    return I2
  
def removeSmallObjects(img, thresh = 200):
    img = skimage.morphology.remove_small_objects(img, thresh).astype(np.uint8)
    return img
  
def removeSmallHoles(img, thresh = 40):
    img = skimage.morphology.remove_small_holes(img, thresh).astype(np.uint8)
    return img
  
def labelImage(img):
    labelledImg = skimage.measure.label(img, background = 255)
    return labelledImg
    
def removeUnsegmentedBlobs(props, areaTh = 40000):
    props1 = []
    props2 = []
    for i in props:
	if i["area"] >= areaTh:
	    props1.append(i)
	else:
	    props2.append(i)
    return props2, props1
    
def removeSmallBlobs(props, areaTh = 500):
    props1 = []
    props2 = []
    for i in props:
	if i["area"] <= areaTh:
	    props1.append(i)
	else:
	    props2.append(i)
    return props2, props1
  
def removeThinBlobs(props, minor_axis_length_Th = 10):
    props1 = []
    props2 = []
    for i in props:
	if i["minor_axis_length"] <= minor_axis_length_Th:
	    props1.append(i)
	else:
	    props2.append(i)
    return props2, props1

def removeSparseBlobs(props):
    centroids = nucleus.get_centroid_list(props)
    props1 = []
    props2 = []
    for i in props:
	bbox = [x for x in i.bbox]
	count = 0
	flag = 0
	for centroid in centroids:
	    if nucleus.checkCentroidInBbox(centroid, bbox) == True:
		count += 1
	    if count >= 2:
		flag = 1
		break  
	if flag == 0:
	    props1.append(i)
	else:
	    props2.append(i)
    return props2, props1
  
def removeSparseBlobs1(props, min_count = 2, mul_factor = 1.0):
    bbox_list = nucleus.get_bounding_box_list(props, (1,1))
    props1 = []
    props2 = []
    for i in props:
	bbox1 = [x for x in i.bbox]
	count = 0
	flag = 0
	for bbox2 in bbox_list:
	    if nucleus.checkOverlapInBbox(bbox1, bbox2, mul_factor) == True:
		count += 1
	    if count >= min_count:
		flag = 1
		break  
	if flag == 0:
	    props1.append(i)
	else:
	    props2.append(i)
    return props2, props1
  

def get_circular_and_elongated_glands(props, ecc_thr1 = 0.90, ecc_thr2 = 0.99):
    props1 = []
    props2 = []
    for i in props:
	if i["eccentricity"] >= ecc_thr1 and i["eccentricity"] <= ecc_thr2:
	    props2.append(i)
	else:
	    props1.append(i)
    return props1, props2

def processBlobs(props):
    totalArea = np.sum([x["area"] for x in props])
    totalNum = len(props)
    areaTh = (totalArea/totalNum)*4.0
    props, big_props = removeUnsegmentedBlobs(props, areaTh)
    #areaTh = (totalArea/totalNum)/4.0
    #props, big_props = removeSmallBlobs(props, areaTh)
    props, sparseprops = removeSparseBlobs1(props)
    props1, props2 = get_circular_and_elongated_glands(props)
    props1.extend(sparseprops)
    props2, sparseprops = removeSparseBlobs1(props2, min_count = 4, mul_factor = 3.0)
    props1.extend(sparseprops)
    return props1, props2
  
def getFeatures_from_props(props):
    ecen_results = processBlobs(props)
    
    totalArea = np.sum([x["area"] for x in ecen_results[0]]) + np.sum([x["area"] for x in ecen_results[1]])
    totalNum = len(ecen_results[0]) + len(ecen_results[1])
    
    features = []
    for group in ecen_results: 
	features.extend([len(group)*10.0/totalNum, np.sum([x["area"] for x in group])*10.0/totalArea, np.sum([x["area"] for x in group])/len(group)])
    features.append(float(totalArea/100000.0))
    features.append(totalNum)
    return features
  
def printFeatures(img):
    props = propsFromImg(img)
    features = getFeatures_from_props(props)
    
    #print "<b>Prop. of no of Nuclie (e < 0.9): </b>", features[0], "<br />"
    #print "<b>Prop. of area of Nuclie (e < 0.9): </b>", features[1], "<br />"
    #print "<b>Avg area of Nuclie (e < 0.9): </b>", features[2], "<br />"
    
    #print "<b>Prop. of no of Nuclie (e > 0.9): </b>", features[3], "<br />"
    #print "<b>Prop. of area of Nuclie (e > 0.9): </b>", features[4], "<br />"
    #print "<b>Avg area of Nuclie (e > 0.9): </b>", features[5], "<br />"
    
    #print "<b>Prop of tile area covered by Nuclie: </b>", features[6], "<br />"
    #print "<b>Total no of Nuclie on tile: </b>", features[7], "<br />"
    
def propsFromImg(img):
    labelled_img, nbr_objects = nucleus.label_image(img)
    props = nucleus.get_label_props(labelled_img)
    return props
  
def markGlands(img, baseImg):
    props = propsFromImg(img)
    props1, props2 = processBlobs(props)
    
    bb_list = nucleus.get_bounding_box_list(props1, (1,1))
    markedImg = nucleus.mark_rectanges(baseImg, bb_list, (0,0,255,255))
    bb_list = nucleus.get_bounding_box_list(props2, (1,1))
    markedImg = nucleus.mark_rectanges(PIL2array(markedImg), bb_list, (0,255,255,255))
    return PIL2array(markedImg), props1, props2

def smoothenContours(contours):
    from scipy.interpolate import splprep, splev
    smoothened = []
    for contour in contours:
	x,y = contour.T
	# Convert from numpy arrays to normal arrays
	x = x.tolist()[0]
	y = y.tolist()[0]
	tck, u = splprep([x,y], s=1.0, per=1)
	u_new = np.linspace(u.min(), u.max(), 25)
	x_new, y_new = splev(u_new, tck, der=0)
	res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
	smoothened.append(np.asarray(res_array, dtype=np.int32))
    return smoothened
  
def findContours(binary, img):
    #binary = binary.astype(np.uint8)
    binary = 255 - binary
    contours, hierarchy = cv2.findContours(binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    #contours = smoothenContours(contours)
    cv2.drawContours(img, contours, -1, (0,0,0), 2)
    return img
  
def segmentNuclei(img):
    im_mask = sp.ndimage.morphology.binary_fill_holes(img < 250)
    min_radius = 5
    max_radius = 25
    imLog = mc.clog(img, im_mask,
	    sigma_min=min_radius * np.sqrt(2),
	    sigma_max=max_radius * np.sqrt(2))
    imNucleiSegMask, Seeds, Max = mc.max_clustering(imLog[0], im_mask, 10)
    imNucleiSegMask = mc.area_open(imNucleiSegMask, 10).astype(np.int)
    return imNucleiSegMask

def overlayNuclei(imNucleiSegMask, imInput):
    # compute nuclei properties
    #objProps = skimage.measure.regionprops(imNucleiSegMask)
    #N = 50
    #HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    #random.shuffle(HSV_tuples);
    #RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    #clrs = RGB_tuples
    #I1 = skimage.color.label2rgb(imNucleiSegMask, imInput, colors=clrs, 
            #bg_label= 0, bg_color = None, alpha = 0.8)
    #I2 = (I1 * 255.9).astype(np.uint8)
    #I2 = array2mat(I2)
    objProps = skimage.measure.regionprops(imNucleiSegMask)
    I1 = skimage.color.label2rgb(imNucleiSegMask, imInput, bg_label= 0, alpha = 0.8)
    I2 = (I1 * 255.9).astype(np.uint8)
    I2 = array2mat(I2)
    return I2

def imageThreshold(img):
    imgSM = img.copy()
    nrows,ncols = imgSM.shape
    m = np.zeros(nrows)
    for i in range(nrows):
        l = imgSM[i, :]
        m1 = l.mean()
        m1 = l[l < m1].mean()
        m[i] = m1
    m1 = m.mean()
    s1 = m.std()
    thr = m1 - 1.5 * s1
    for i in range(nrows):
        l = imgSM[i, :]
        imgSM[i, l > thr] = 255
    return imgSM

def getGeneExpression(imgName, genelist):
    expData = pd.DataFrame.from_csv('/booleanfs2/sahoo/Data/Piero/Colon/tcga-2017-m-expr.txt', sep='\t', header=0, index_col=None)
    names = [x.split(":")[0] for x in expData["Name"]]
    expData = expData.set_index([names])
    hyphen_index = [x.start() for x in re.finditer("-", imgName)]
    for i in list(expData):
	if imgName[:hyphen_index[2]] in i:
	    #print "<b>", "Patient ID: ", "</b>", i
	    #for gene in genelist:
		#print "<b>", gene, ": ", "</b>", expData.loc[gene, i], "&nbsp;&nbsp;"
	    #print "<br />"
    
def PIL2array(img):
    return np.array(img.getdata(),
	    np.uint8)[:, :3].reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def array2base64_old(arr):
    fig = pl.figure(figsize=(6,6))
    fig.add_axes([0, 0, 1, 1])
    pl.imshow(arr)
    pl.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0);
    b64 = base64.encodestring(buf.getvalue())
    return b64

def array2base64(arr):
    return mat2base64(array2mat(arr))

def array2mat(arr):
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def mat2base64(mat):
    """Ecodes image array to Base64"""
    encoded = cv2.imencode(".png", mat)[1]
    b64 = base64.encodestring(encoded)
    return b64

def base642mat(img_string):
    """Decodes Base64 string to an image array"""
    first_coma = img_string.find(',')
    img_bytes = base64.decodestring(img_string[first_coma:])
    image = np.asarray(bytearray(img_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def reinhard_mat(img):
    refimg = cv2.imread("L1.png")
    im_lab = cv2.cvtColor(refimg,cv2.COLOR_BGR2LAB)
    target_mu = np.zeros(3)
    target_sigma = np.zeros(3)
    src_mu = np.zeros(3)
    src_sigma = np.zeros(3)

    for i in range(3):
        target_mu[i] = im_lab[:, :, i].mean()
        target_sigma[i] = (im_lab[:, :, i] - target_mu[i]).std()

    im_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    for i in range(3):
        src_mu[i] = im_lab[:, :, i].mean()
        src_sigma[i] = (im_lab[:, :, i] - src_mu[i]).std()

    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i] * target_sigma[i] + target_mu[i]

    im_norm = cv2.cvtColor(im_lab,cv2.COLOR_LAB2BGR)
    return im_norm

def reinhard_array(img):
    refimg = skimage.io.imread("L1.png")[:, :, :3]
    im_lab = mc.rgb_to_lab(refimg)
    target_mu = np.zeros(3)
    target_sigma = np.zeros(3)
    src_mu = np.zeros(3)
    src_sigma = np.zeros(3)

    for i in range(3):
        target_mu[i] = im_lab[:, :, i].mean()
        target_sigma[i] = (im_lab[:, :, i] - target_mu[i]).std()

    im_lab = mc.rgb_to_lab(img)
    for i in range(3):
        src_mu[i] = im_lab[:, :, i].mean()
        src_sigma[i] = (im_lab[:, :, i] - src_mu[i]).std()

    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i] * target_sigma[i] + target_mu[i]

    im_norm = mc.lab_to_rgb(im_lab)
    im_norm[im_norm > 255] = 255
    im_norm[im_norm < 0] = 0
    im_norm = im_norm.astype(np.uint8)

    return im_norm

def color_deconvolution(im_rgb, w, I_0=None):
    return mc.color_deconvolution(im_rgb, w, I_0)

def detectEdges(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    edges = auto_canny(blurred)
    edges[edges == 0] = img[edges == 0]
    return edges

def detectBlobs(img):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    ## Change thresholds
    #params.minThreshold = 10;
    #params.maxThreshold = 200;
    ## Filter by Area.
    #params.filterByArea = True
    #params.minArea = 1500
    ## Filter by Circularity
    #params.filterByCircularity = True
    #params.minCircularity = 0.1
    ## Filter by Convexity
    #params.filterByConvexity = True
    #params.minConvexity = 0.87
    ## Filter by Inertia
    params.filterByInertia = True
    params.maxInertiaRatio = 0.0001
    detector = cv2.SimpleBlobDetector(params)
    # Detect blobs.
    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints,\
      np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    #edged = cv2.Canny(image, lower, upper)
    edged = cv2.Canny(image, 0, 255)

    # return the edged image
    return edged

def connectedComponents(img):
    img = img.astype(np.uint8)
    out = np.zeros(img.shape)
    contours, hierarchy = cv2.findContours(img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(out, contours, i, (i), cv2.cv.CV_FILLED);
    return len(contours), out

def getHematoxylin(img):
    # create stain to color map
    stainColorMap = {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin':       [0.07, 0.99, 0.11],
            'dab':         [0.27, 0.57, 0.78],
            'null':        [0.0, 0.0, 0.0]
            }

    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains

    # create stain matrix
    W = np.array([stainColorMap[stain_1], 
        stainColorMap[stain_2], 
        stainColorMap[stain_3]]).T

    # perform standard color deconvolution
    imDeconvolved = color_deconvolution(img, W)
    H = imDeconvolved.Stains[:, :, 0]
    return H

def getBlueRatio(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    BR = ((100.*B)/(1.+R+G))*(256./(1.+B+R+G))
    #BR[BR > 100] = 100
    m = np.percentile(BR, 99)
    BR[BR > m] = m
    I1 = BR
    I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
    I2 = 255 - I2
    return I2
