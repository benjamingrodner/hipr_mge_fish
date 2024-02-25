# Python module
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Contains functions to assign spots segmentation to cell segmentation
"""
# =============================================================================

from skimage.feature import peak_local_max
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# =============================================================================


def _get_merged_peaks(im, min_distance=5):
    is_peak = peak_local_max(im, indices=False,
                             min_distance=min_distance) # outputs bool image
    labels = label(is_peak)[0]
    merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
    return np.array(merged_peaks)


def group_spot_maxima_by_spot_id(raw, seg, min_distance=5):
    # Get local maxima
    maxs = _get_merged_peaks(raw, min_distance=min_distance)
    # Assign local maxima to spots
    max_ids = [seg[int(m[0]),int(m[1])] for m in maxs]
    intensities = [raw[int(m[0]),int(m[1])] for m in maxs]
    maxs_ = np.concatenate((maxs, np.array([intensities, max_ids]).T),
                                axis=1).tolist()
    # Group maxs by spot id
    maxs_.sort(key=itemgetter(3))
    iter = groupby(maxs_, itemgetter(3))
    maxs_grouped = [[int(key), [v[:3] for v in vals]] for (key, vals) in iter]
    # maxs_grouped = [[id,[(m[0],m[1],m[3]) for m in list(maxs_) if m[2]==id]]
    #                 for id in ids]
    return pd.DataFrame(maxs_grouped, columns=['label', 'maxs_loc_int'])


def get_local_max_props(raw, seg, min_distance=5):
    maxs = _get_merged_peaks(raw, min_distance=min_distance)
    # Assign local maxima to spots
    max_ids = [seg[int(m[0]),int(m[1])] for m in maxs]
    intensities = [raw[int(m[0]),int(m[1])] for m in maxs]
    maxs_ = np.concatenate((maxs, np.array([intensities, max_ids]).T),
                                axis=1).tolist()
    maxs_.sort(key=itemgetter(3))
    return pd.DataFrame(maxs_, columns=['row','col','intensity','spot_id'])



def maxs_to_cell_id(maxs, cell_seg, search_radius):
    cell_seg_pad = np.pad(cell_seg, search_radius, mode='edge')
    maxs_arr = maxs.loc[:,['row','col']].values
    window = 2*search_radius + 1
    max_cid_list = []
    # for each max location
    for i, (r, c) in enumerate(maxs_arr):
        r,c = int(r),int(c)
        cid = cell_seg[r,c]
        # Check if the max is in a cell seg
        if cid:
            dist = 0  # if so distance is zero
        else:
            # get a slice of the seg in a range around the max
            slice = cell_seg_pad[r:r+window, c:c+window]
            cid = 0 # Set non-cell associated values
            dist = 1e5
            for j in range(window):  # iterate over the slice
                for k in range(window):
                    cid_ = slice[j,k]
                    if not cid_:  # Ignore zeros
                        continue
                    else:
                        # calculate the distance to the center of the slice
                        dist_ = ((search_radius-j)**2 + (search_radius-k)**2)**(1/2)
                        # Restrict to a circle (ignore the corners of the slice)
                        if dist_ > search_radius:
                            continue
                        # Replace the distance and cell id values if closer
                        elif dist_ < dist:
                            dist = dist_
                            cid = cid_
        max_cid_list.append([i,cid,dist])  # write max id, cell_id, distance
        max_cid_df = pd.DataFrame(max_cid_list, columns=['max_id','cell_id','cell_dist'])
    return pd.concat([maxs, max_cid_df], axis=1)  # add to max props


def convolve_bool_circle(arr, r=10):
    d = 2*r + 1
    # get circle of ones
    k = np.ones((d,d))
    for i in range(d):
        for j in range(d):
            dist = ((r-i)**2 + (r-j)**2)**(1/2)
            if dist > r: k[i,j] = 0
    conv = np.pad(arr, r).astype(np.double)
    return convolve(conv, k) > 0


def get_cell_spot_maxs(cell_props, seg, spot_raw, r=10):
    # get bounding box values
    bboxes_ids = []
    for i, row in cell_props.iterrows():
        bb = (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'], row['label'])
        bboxes_ids.append([int(b) for b in bb])
    spot_max_int = []
    # Get each cell
    for bb in bboxes_ids:
        # Extract bounding box make boolean mask
        mask = seg[bb[0]:bb[2], bb[1]:bb[3]] == bb[4]
        # convolve to expand mask edges
        mask_conv = convolve_bool_circle(mask, r)
        # Extract bounding box from spot image
        sr_pad = np.pad(spot_raw, r,).astype(np.double)
        sr_bbox = sr_pad[bb[0]: bb[2] + 2*r, bb[1]: bb[3] + 2*r]
        sr_bbox_masked = sr_bbox * mask_conv
        spot_max_int.append(np.max(sr_bbox_masked))
    return spot_max_int


def get_spots_with_multimax(spot_seg, local_maxima):
    # %% codecell
    # Get spots in cells
    sscs = spot_seg
    # get maxs in spots
    mss = local_maxima*(sscs>0)
    # Mask spots with maxs
    s_m = sscs*(local_maxima>0)
    # Make 3d array with maxs and masked spots
    smarr = np.dstack((s_m, mss))
    # Get unique on axis 2
    # s_ = (0,100,0,100)
    # smarr_ = smarr[s_[0]: s_[1],s_[2]:s_[3],:]
    # m_ = mss[s_[0]: s_[1],s_[2]:s_[3]] > 0
    sm = np.unique(smarr[mss > 0], axis=0)
    # sm
    # plt.imshow(mss[s_[0]: s_[1],s_[2]:s_[3]] > 0, interpolation='None')
    # plt.imshow(sscs[s_[0]: s_[1],s_[2]:s_[3]]>0, interpolation='None')
    # get counts on the spot column
    s_c = np.unique(sm[:,0], return_counts=True)
    # get all spots with count < 1
    s_c_slim = s_c[0][s_c[1] > 1]
    # return a table with the spot ids with mltiple max ids
    sm_slim = [i for i in sm if i[0] in s_c_slim]
    return pd.DataFrame(sm_slim, columns=['spot_id','max_id'])


def generate_voronoi_diagram(width, height, centers_x, centers_y):
    # Create grid containing all pixel locations in image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # Find squared distance of each pixel location from each center: the (i, j, k)th
    # entry in this array is the squared distance from pixel (i, j) to the kth center.
    squared_dist = (x[:, :, np.newaxis] - centers_x[np.newaxis, np.newaxis, :]) ** 2 + \
                   (y[:, :, np.newaxis] - centers_y[np.newaxis, np.newaxis, :]) ** 2
    # Find closest center to each pixel location
    return np.argmin(squared_dist, axis=2) + 1  # Array containing index of closest center

from skimage.segmentation import relabel_sequential


def split_multimax_spots(multimax, spot_props, seg, max_props, raw):
    seg_new = seg.copy()
    # subset only multimax spots
    spots = multimax.spot_id.unique()
    spot_props_multi = spot_props[spot_props.label.isin(spots)]
    # iterate through
    for index, row in spot_props_multi.iterrows():
        # Extract bounding box of spot seg
        bb = (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'], row['label'])
        bb = [int(j) for j in bb]
        seg_bb = seg[bb[0]:bb[2], bb[1]:bb[3]]
        mask = seg_bb == bb[4]
        # Subset max props
        maxs = multimax.loc[multimax.spot_id == bb[4], 'max_id']
        max_props_sub = max_props[max_props.label.isin(maxs)]
        # print(max_props_sub)
        # Get max indices
        centers_x = max_props_sub['centroid-1'].to_numpy() - bb[1]
        centers_y = max_props_sub['centroid-0'].to_numpy() - bb[0]
        # get voroni diagram
        height, width = mask.shape
        vor = generate_voronoi_diagram(width, height, centers_x, centers_y)
        # mask voronoi
        vor_seg = vor * mask
        # get max of current spot id list
        sid = np.max(seg_new) + 1
        # assign other regions to new spot id
        for i in range(1,len(centers_x)+1):
            # print(sid)
            vor_seg[vor_seg == i] = sid
            sid += 1
        # mask out old spot in spot seg image
        seg_bbox = seg_bb
        seg_bbox_masked = seg_bbox * (mask == 0)
        # add in new seg to the seg image
        new_seg_bbox = seg_bbox_masked + vor_seg
        seg_new[bb[0]:bb[2], bb[1]:bb[3]] = new_seg_bbox
    return seg_new



def assign_spots_to_cells(spot_seg, cell_seg):
    # Get spots in cells
    sscs = spot_seg*(cell_seg>0)
    # Mask cells with spots
    cs_masked = cell_seg*(sscs>0)
    # Make 3d array with maxs and masked spots
    stack = np.dstack((cs_masked, sscs))
    # Get unique on axis 2
    # s_ = (0,1000,0,1000)
    # stack_ = stack[s_[0]: s_[1],s_[2]:s_[3],:]
    # sscs_ = sscs[s_[0]: s_[1],s_[2]:s_[3]] > 0
    # assgn = np.unique(stack_[sscs_ > 0], axis=0)
    assgn = np.unique(stack[sscs > 0], axis=0)
    return pd.DataFrame(assgn, columns=['cell_id','spot_id'])


def assign_multimapped_spots(spot_cid_df, spot_props, spot_raw, cell_seg):
    # Subset spot props to only thos that multimap
    val_counts = spot_cid_df.spot_id.value_counts()
    sids = val_counts[val_counts>1].index
    spot_props_sub = spot_props.loc[spot_props.label.isin(sids)]
    spot_cid_df_new = spot_cid_df.copy()
    # iterate through
    for index, row in spot_props_sub.iterrows():
        # get bounding box for spot
        bb = (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'], row['label'])
        bb = [int(j) for j in bb]
        sr_bbox = spot_raw[bb[0]:bb[2], bb[1]:bb[3]]
        # get spot bbox for cell seg
        cs_bbox = cell_seg[bb[0]:bb[2], bb[1]:bb[3]]
        # get a list of indices for each pixel of each cell in bbox
        cids = np.unique(cs_bbox)
        if cids.shape[0] == 1:
            if cids[0] == 0:
                cid = 0
            else:
                cid = cids[0]
        else:
            arr = np.empty((0,2))
            for c in cids[1:]:
                arr = np.concatenate([arr, np.argwhere(cs_bbox==c)], axis=0)
            y = arr[:,0]
            x = arr[:,1]
            # Find the index of the spot maximum
            m = np.argwhere(sr_bbox == np.max(sr_bbox))
            # Find the nearest neighbor pixel and get its cell id
            dists = (x - m[0,1]) ** 2 + (y - m[0,0]) ** 2
            # if arr.shape[0] == 0: print(cids, arr, spot_cid_df[spot_cid_df.spot_id == row.label],
            #                         np.argwhere(cs_bbox==cids[0]))
            ind_min = arr[np.argwhere(dists == np.min(dists)),:].astype(int)
            cid = cs_bbox[ind_min[0,0,0],ind_min[0,0,1]]
        # assign that cell id to the spot props table and remove other assignments
        sid = int(row.label)
        scdfnew = pd.DataFrame({'cell_id':[cid], 'spot_id':[sid]})
        spot_cid_df_new.drop(spot_cid_df_new[spot_cid_df_new['spot_id'] == sid].index,
                                                             inplace = True)
        spot_cid_df_new = spot_cid_df_new.append(scdfnew)
    return spot_cid_df_new


def filter_seg_objects(seg, props, filter):
    seg_new = seg.copy()
    remove_cids = props.loc[props[filter] == 0, ['label','bbox']].values
    for i, (c, b) in enumerate(remove_cids):
        b = eval(b)
        b_sub = seg_new[b[0]:b[2],b[1]:b[3]]
        b_sub = b_sub * (b_sub != c)
        seg_new[b[0]:b[2],b[1]:b[3]] = b_sub
    return seg_new



# TODO rewrite max props so that it derives from the labeled max array
