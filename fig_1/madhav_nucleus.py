import cv2
import numpy as np
import scipy.stats as stats
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
import argparse, json
import math, csv
import os, sys, re
import cPickle as pickle
from PIL import Image, ImageDraw
import openslide as ops
from skimage.color import rgb2hed
from skimage import morphology as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_niblack
from matplotlib.colors import LinearSegmentedColormap
import ColorSegmentation
import stepminer as stepm

DITHER_THRES = 200
MIN_INTENSITY = 0
MAX_INTENSITY = 255

TILE_LENGTH = 1000 # Along x axis
TILE_WIDTH = 1000 # Along y axis

TILE_NUCLEUS_THRES = 100
# block size for ni balck threshold
block_size = 31

genename = 'MUC2:'

def typical_obj_size(objList,method='avg'):
    if method == 'med':
        tpc_width = np.median([abs(obj[3]-obj[1]) for obj in objList])
        tpc_height = np.median([abs(obj[2]-obj[0]) for obj in objList])
        return [tpc_width,tpc_height]
    else:
        tpc_width = np.mean([abs(obj[3]-obj[1]) for obj in objList])
        tpc_height = np.mean([abs(obj[2]-obj[0]) for obj in objList])
        return [tpc_width,tpc_height]
        
def remove_tiny_objs(objprops):
	total_area = 0
	for obj in objprops:
		total_area += obj["area"]
	tpc_area = total_area/ len(objprops)
	return [obj for obj in objprops if obj["area"] >= tpc_area]
  
def extract_roi(img, level):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#print "gray converted"
	dither_img = naive_dithering(gray)
	im_filled = morphology.binary_closing(dither_img, np.ones((3,3)),iterations=1)
	#print "binary closing done"
	labelled_img, nbr_objects = label_image(im_filled)
	objprops = get_label_props(labelled_img)
	objprops = postprocess(objprops)
	roi_list = get_bounding_box_list(objprops, level)
	return roi_list

def postprocess(objprops):
	objprops = remove_tiny_objs(objprops)
	return objprops

def naive_dithering(gray_img,dither_thres=DITHER_THRES,inv=True):
	# input: an image
	# output: a black-white image reflecting the same content
	logical_idx = (gray_img <= dither_thres) if inv else (gray_img > dither_thres)
	output_img = MAX_INTENSITY*logical_idx
	return output_img

def label_image(filled_img):
	labelled_img , nbr_objects = label(filled_img, neighbors = 4, return_num = True, background = 255)
	return labelled_img, nbr_objects

def label_cores(filled_img, level):
	labelled_img, nbr_objects = label_image(filled_img)
	props = get_label_props(labelled_img)
	return get_bounding_box_list(props, level)

def get_label_props(labelled_img):
	props = regionprops(labelled_img)
	return props

def get_bounding_box_list(props, level):
	box_list = []
	for i in range(len(props)):
		temp = [x for x in props[i].bbox]
		temp[0] = int(temp[0] * level[1])
		temp[1] = int(temp[1] * level[0])
		temp[2] = int(temp[2] * level[1])
		temp[3] = int(temp[3] * level[0])
		#bbox_tuple = tuple(temp)
		box_list.append(temp)
	return box_list

def get_centroid_list(props):
	centroid_list = []
	for i in range(len(props)):
		temp = [x for x in props[i].centroid]
		centroid_tuple = tuple(temp)
		centroid_list.append(centroid_tuple)
	return centroid_list
      
def get_distance_matrix(centroids):
	distance_matrix = [[2000000 for i in range(len(centroids))] for j in range(len(centroids))]
	for i in range(len(centroids)):
	    for j in range(len(centroids)):
		if i != j:
		    distance_matrix[i][j] = ((centroids[i][0]-centroids[j][0])**2) + ((centroids[i][1]-centroids[j][1])**2)
	return distance_matrix
	
def get_minimum_dist_for_nuclie(centroids):
	distance_matrix = get_distance_matrix(centroids)
	minimum_arr = [0 for i in range(len(centroids))]
	for i in range(len(minimum_arr)):
		minimum_arr[i] = min(distance_matrix[i])
	return minimum_arr
      
def checkCentroidInBbox(centroid, bbox):
	side = max((bbox[2] - bbox[0]),(bbox[3] - bbox[1]))
	height = 0.5 * (side - (bbox[2] - bbox[0]))
	width = 0.5 * (side - (bbox[3] - bbox[1]))
	if centroid[0] >= bbox[0] and centroid[0] <= bbox[2] and centroid[1] >= bbox[1] and centroid[1] <= bbox[3]:
		return False
	if centroid[0] >= ((bbox[0]-height)-(side/2.0)) and centroid[0] <= ((bbox[2]+height)+(side/2.0)) \
	  and centroid[1] >= ((bbox[1]-width)-(side/2.0)) and centroid[1] <= ((bbox[3]+width)+(side/2.0)):
		return True
	else:
		return False

def checkOverlapInBbox(bbox1Temp, bbox2Temp, mul_factor = 1.):
	bbox1 = list(bbox1Temp)
	bbox2 = list(bbox2Temp)
	for bbox in [bbox1, bbox2]:
		side = max((bbox[2] - bbox[0]),(bbox[3] - bbox[1]))
		height = 0.5 * (side - (bbox[2] - bbox[0]))
		width = 0.5 * (side - (bbox[3] - bbox[1]))
		bbox[0] += height
		bbox[1] += width
		bbox[2] += height
		bbox[3] += width
		bbox[0] -= (mul_factor - 1)*(side/2.)
		bbox[1] -= (mul_factor - 1)*(side/2.)
		bbox[2] += (mul_factor - 1)*(side/2.)
		bbox[3] += (mul_factor - 1)*(side/2.)
	#If one rectangle is on left side of other
	if (bbox1[1] > bbox2[3] or bbox2[1] > bbox1[3]):
		return False
	#If one rectangle is above other
	if (bbox1[0] > bbox2[2] or bbox2[0] > bbox1[2]):
		return False
	return True

def get_nucleus_props(img_ng = None, level = 0):
	labelled_img, nbr_objects = label_image(img_ng)
	props = get_label_props(labelled_img)
	return props
      
def count_glands(img, level = (1,1)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#print "gray converted"
	dither_img = naive_dithering(gray)
	im_filled = morphology.binary_closing(dither_img, np.ones((3,3)),iterations=1)
	#print "binary closing done"
	labelled_img, nbr_objects = label_image(im_filled)
	props = get_label_props(labelled_img)
	return get_bounding_box_list(props, level)
	
def show_objs(objList,image,color=(255,0,0),linewidth=2,showWindow=True,waitTime=0):
	for obj in objList:
		if obj:
			x_start = obj[0]
			x_end = obj[1]
			y_start = obj[2]
			y_end = obj[3]
			cv2.rectangle(image,(x_start,y_start),(x_end,y_end),color,linewidth)
	if showWindow:
		cv2.imshow("show",image)
		cv2.waitKey(waitTime)

def color_segmentation(img, saveFlag = 0):
	ihc_hed = rgb2hed(img)
	if saveFlag:
		cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
		cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white','saddlebrown'])
		cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet','white'])
		plt.imsave('hema.jpg', ihc_hed[:, :, 0], cmap=cmap_hema)
		plt.imsave('eosin.jpg', ihc_hed[:, :, 1], cmap=cmap_eosin)
		plt.imsave('dab.jpg', ihc_hed[:, :, 2], cmap=cmap_dab)
	#print "color channels done"
	return ihc_hed

def mark_rectanges(img, coord_list, outlineColor):
	marked = Image.fromarray(img)
	pdraw = ImageDraw.Draw(marked)
	for i in coord_list:
		pdraw.rectangle([(i[1],i[0]),(i[3],i[2])], outline = outlineColor, fill = None)
	return marked

def edge_detect_ng1(img):
	KM_Results = ColorSegmentation.KM_nuclei_segment(img)
	
	I2 = cv2.cvtColor(KM_Results[2], cv2.COLOR_BGR2GRAY)
	
	from skimage.feature import canny
	edges = canny(I2/255.)
	
	I1 = edges
	I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	Image.fromarray(I2).save("edges.jpg")
	
	#from scipy import ndimage as ndi
	#fill_coins = ndi.binary_fill_holes(edges)
	#I1 = fill_coins
	#I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	#Image.fromarray(I2).save("edges_filled.jpg")
	
def segment_using_stepminer(img):
	stepm_results = remove_backgrd_stepm(img)
	cv2.imwrite('img_light.jpg',stepm_results[0])
        cv2.imwrite('img_dark.jpg',stepm_results[1])
        
	#img_temp = stepm_results[1].copy()

        #blobs = []
        #blobs = count_glands(stepm_results[1])
        #print "No of blobs :", len(blobs)
        
	#if len(blobs) > 0:
		#blob_sizes = []
		#for blob in blobs:
			#blob_sizes.append((blob[3]-blob[1]) + (blob[2]-blob[0]))
		
		#blob_sizes = sorted(blob_sizes)
	
		#thresh = (stepm.fitstep(blob_sizes)["threshold"])
		#print "thresh =", thresh 
	
		#for blob in blobs:
			#if blob[3]-blob[1] <= 5 or blob[2]-blob[0] <= 5:
				#continue
			#elif (((blob[3]-blob[1]) + (blob[2]-blob[0])) > thresh):
				#img_section = img_temp[blob[0]:blob[2],blob[1]:blob[3],:]			
				#stepm_section_results = remove_backgrd_stepm(img_section, stepm_results[2])
				#for i in range(blob[3] - blob[1]):
					#for j in range(blob[2] - blob[0]):
						#img_temp[blob[0]+j,blob[1]+i,:] = stepm_section_results[1][j,i,:]
        
	#cv2.imwrite('image_dark2.jpg',img_temp)
	
def segment_nucleus(img):
	KM_Results = ColorSegmentation.KM_nuclei_segment(img)
        k = KM_Results[0]
        cv2.imwrite('bg1.jpg',KM_Results[1])
        cv2.imwrite('ng1.jpg',KM_Results[2])
        if k >2:
		cv2.imwrite('ps1.jpg',KM_Results[3])
	
	img_temp = KM_Results[2].copy()
	
	stats = 50
	rounds  =  0

	while stats < 2000 and rounds <= 1:
		rounds += 1 
		blobs = []
		blobs = count_glands(img_temp)

		if len(blobs) == 0:
			break
		print "No of glands", len(blobs)
		blob_sizes = []
		for blob in blobs:
			blob_sizes.append((blob[3]-blob[1]) + (blob[2]-blob[0]))

		blob_sizes = sorted(blob_sizes)
	
		stats = stepm.fitstep(blob_sizes)["statistic"]
		print "Stats =" , stats
		thresh = stepm.fitstep(blob_sizes)["threshold"]
		print "thresh =", thresh 
	
		flag = 0
		for blob in blobs:
			if blob[3]-blob[1] <= 5 or blob[2]-blob[0] <= 5:
				continue
			elif (((blob[3]-blob[1]) + (blob[2]-blob[0])) > thresh):
				img_section = img_temp[blob[0]:blob[2],blob[1]:blob[3],:]
				KM_Results = ColorSegmentation.KM_nuclei_segment(img_section)
				for i in range(blob[3] - blob[1]):
					for j in range(blob[2] - blob[0]):
						img_temp[blob[0]+j,blob[1]+i,:] = KM_Results[2][j,i,:]
				flag = 1
		if flag == 0:
			print "Flag break"
			break
			
	cv2.imwrite('ng2_2000.jpg',img_temp)
	print "No of rounds " , rounds
	ihc_hed = color_segmentation(img_temp, 0)
	#Using only the Hematoxin channels
	I1 = ihc_hed[:, :, 0]
	I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	Image.fromarray(I2).save("hema_grey.jpg")

	from skimage.feature import canny
	edges = canny(I2/255.)
	
	I1 = edges
	I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	Image.fromarray(I2).save("edges.jpg")
	
	from scipy import ndimage as ndi
	fill_coins = ndi.binary_fill_holes(edges)
	I1 = fill_coins
	I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	Image.fromarray(I2).save("edges_filled.jpg")
	
	adaptive_thresh = threshold_niblack(I2, block_size)
	
	I1 = (I2 > adaptive_thresh.max()).astype(int)
	I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	Image.fromarray(I2).save("binary.jpg")
	
	I1 = morphology.binary_opening(I2, np.ones((2,2)),iterations=1)
	I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	Image.fromarray(I2).save("opened.jpg")

	I1 = morphology.binary_closing(I2, np.ones((2,2)),iterations=1)
	I2 = (((I1 - I1.min()) / (I1.max() - I1.min())) * 255.9).astype(np.uint8)
	Image.fromarray(I2).save("closed.jpg")
	
	print "Counting Nucleus"
	nucleusProps = []
	nucleusList = []
	nucleusProps = get_nucleus_props(I2)
	nucleusList = get_bounding_box_list(nucleusProps, level = (1,1))
	print "Number of nucleus:", len(nucleusList)
	return nucleusProps, nucleusList