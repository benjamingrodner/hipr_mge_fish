# Functions for processing rois drawn manually to connect regions between MGEFISh and HiPRFISH

## Imports

from skimage.segmentation import find_boundaries
from scipy.ndimage import convolve
from PIL import Image
import numpy as np


## Functions

def load_tiff(fn):
    return array(Image.open(fn))

def convolve_bool_circle(arr, r=10):
    '''
    Widen the mask 'arr' by radius 'r'
    '''
    d = 2*r + 1
    # get circle of ones
    k = np.ones((d,d))
    for i in range(d):
        for j in range(d):
            dist = ((r-i)**2 + (r-j)**2)**(1/2)
            if dist > r: k[i,j] = 0
    conv = np.pad(arr, r).astype(np.double)
    return convolve(conv, k) > 0


def get_thick_roi_outlines(roi_npy_fns, im_shape, mge_roi_props_df, r=5):
    mge_roi_overlay = np.zeros(im_shape)
    for fn in roi_npy_fns:
        # Load roi and props
        roi = 1*(np.load(fn) > 0)
        row = mge_roi_props_df.iloc[i,:]
        bbox = eval(row['bbox'])
        roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        roi_line = find_boundaries(roi_box, mode='thick')
        roi_line_thick = convolve_bool_circle(roi_line, r=r)[r:-r,r:-r]
        overl_box = mge_roi_overlay[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        overl_box = overl_box + roi_line_thick
        mge_roi_overlay[bbox[0]:bbox[   2],bbox[1]:bbox[3]] = overl_box
    return 1*(mge_roi_overlay > 0)


def sumdist(shift, pts1, pts2):
    pts2 = np.copy(pts2)
    for i, s in enumerate(shift):
        pts2[:,i] = pts2[:,i] + s
    D = np.sqrt(np.sum((pts1-pts2)**2, axis=1))
    return(np.sum(D), pts2)


# Register sister points on two images, points must be in order
def register_points_mindist(pts1, pts2, n=100, min_diff=2, step_reduce = 0.75):
    '''
    pts1/pts2: Arrays (n,2) of n sister points, arrays must be in the same order
    n: (integer,) maximum number of iterations allowed
    min_diff: (float,) if the change in summed distance between sister points is less than this value, then the current shift value is selected
    step_reduce: (float,) < 1 if the current step is too large, the step size is multiplied by this value
    '''
    val_init = sumdist((0,0), pts1, pts2)[0]
    diff=1e10
    step = np.min(np.abs(pts1-pts2))
    val_old = val_init
    shift_list = []
    i = 0
    while (i < n) and (diff > min_diff):
        shifts = [[0,step],[0,-step],[step,0],[-step,0]]  # Rook movement options
        sd_pts = [sumdist(s, pts1, pts2) for s in shifts]  # evaluate moves
        sd = [s[0] for s in sd_pts]  # Extract values
        pts_all = [s[1] for s in sd_pts]  # Extract new points
        sd_ind = np.argmin(sd)  # Evaluate options
        val_new = sd[sd_ind]
        shift_choice = shifts[sd_ind]
        if val_new < val_old:  # Compare and replace values
            pts2 = pts_all[sd_ind]
            shift_list.append(shift_choice)
            diff = val_old - val_new
            val_old = val_new
        else:  # ...OR reduce step size
            step = round(step*step_reduce)
        i += 1

    shift_final = np.sum(np.array(shift_list), axis=0).astype(int)
    return shift_final
