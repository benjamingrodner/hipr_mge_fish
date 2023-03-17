
"""
Hao Shi 2019
De Vlaminck Lab
Cornell University
"""

import os
import re
import sys
import glob
import yaml
import joblib
import skimage
import argparse
import numpy as np
import pandas as pd
from time import time
from skimage import measure
from sklearn import neighbors as nb

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################



def main():


    parser = argparse.ArgumentParser('Classify single cell spectra')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-sf', '--seg_fn', dest = 'seg_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-rf', '--reg_fn', dest = 'reg_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-cf', '--classif_fn', dest = 'classif_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Get classifier format
    ref_train_dir = config['reference_training']['out_dir']
    spc = config['ref_train_simulations']
    probe_design_filename = (config['__default__']['PROBE_DESIGN_DIR'] +
                                '/' + config['probe_design_filename']
                                )
    probe_design_basename = os.path.splitext(os.path.basename(probe_design_filename))[0]
    ref_clf = config['pkl_fmt'].format(ref_train_dir, spc, probe_design_basename, '{}')

    # Load input spectra
    props = pd.read_csv(args.seg_props)
    print(args.seg_props, props.columns)

    # Get cell spectra
    avgint_cols = [str(i) for i in range(config['chan_start'],config['chan_end'])]
    avgint = props[avgint_cols]


    # Get rough classification
    n_las = config['n_lasers']
    n_chan = avgint.shape[1]
    avgint_norm = avgint.values/np.max(avgint.values, axis = 1)[:,None]
    classarr = np.zeros((avgint_norm.shape[0], n_las))
    avgint_norm = np.concatenate((avgint_norm, classarr), axis = 1)
    for i in range(n_las):
        las_params = config['rough_classifier']['laser_' + str(i)]
        chans = las_params['channels']
        llim = las_params['lower_lim']
        avgint_norm[:, n_chan + i] = (
                np.max(avgint_norm[:,chans], axis = 1) > llim
            )*1
    rc_colnames = ['rough_class_' + str(i) for i in range(n_las)]
    props[rc_colnames] = avgint_norm[:,-n_las:]


    # Get umap classifier
    umap_transform = joblib.load(ref_clf.format('umap_transform'))
    umap_transform_embedding = umap_transform.embedding_
    clf_umap = joblib.load(ref_clf.format('svc'))
    # Run classification iteratively
    n = config['n_classif_iters']
    cell_ids_norm_arr = np.zeros((props.shape[0], n))
    nn_dist_arr = np.zeros((props.shape[0], n))
    for r in range(n):
        # Get nearest neighbors for cell spectra
        t0 = time()
        nbrs = nb.NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(umap_transform_embedding)
        t1 = time()
        if r == 2: print('t1:',t1-t0)
        umap_transform_embeding_new = umap_transform.transform(avgint_norm)
        t2 = time()
        if r == 2: print('t2:',t2-t1)
        nn_dists, nn_indices = nbrs.kneighbors(umap_transform_embeding_new)
        t3 = time()
        if r == 2: print('t3:',t3-t2)
        nn_dist_arr[:,r] = np.min(nn_dists, axis = 1)

        # Classify nearest neighbor spectra
        avgint_umap_nn = umap_transform.embedding_[nn_indices[:,0],:]
        cell_ids_norm = clf_umap.predict(avgint_umap_nn)
        t4 = time()
        if r == 2: print('t4:',t4-t3)
        cell_ids_norm_arr[:,r] = cell_ids_norm

    # Pick the classification with the smallest distance to the training set
    nn_barcode = []
    # print([np.unique(nn_dist_arr[i,:], return_counts=True) for i in range(nn_dist_arr.shape[0])])
    nn_indices = np.argmin(nn_dist_arr, axis = 1)
    nn_dists_min = np.min(nn_dist_arr, axis=1)
    for b in range(cell_ids_norm_arr.shape[0]):
        nn_barcode.append(cell_ids_norm_arr[b,nn_indices[b]])
    props['cell_barcode'] = nn_barcode
    props['nn_dist'] = nn_dists_min


    # Get spectral centroid distance for later evaluation of undersegmentation
    image_seg = np.load(args.seg_fn)
    image_registered = np.load(args.reg_fn)
    spectral_centroid_distance = []
    for i in range(props.shape[0]):
        barcode = props.loc[i, 'cell_barcode']
        cell_label = props.loc[i, 'label']
        cell_index = np.where(image_seg == cell_label)
        cell_pixel_intensity = image_registered[image_seg == cell_label, :]
        cx = np.average(cell_index[0])
        cy = np.average(cell_index[1])
        cxs = cell_pixel_intensity*cell_index[0][:,None]
        cys = cell_pixel_intensity*cell_index[1][:,None]
        cpi_av = np.average(cell_pixel_intensity, axis = 0)
        cx_spectral = np.average(cxs, axis = 0) / cpi_av
        cy_spectral = np.average(cys, axis = 0) / cpi_av
        scd = np.sqrt((cx - cx_spectral)**2 + (cy - cy_spectral)**2)
        spectral_centroid_distance.append(np.median(scd))
    props['spectral_centroid_distance'] = spectral_centroid_distance
    props.to_csv(args.classif_fn, index = None)


    return

if __name__ == '__main__':
    main()






#####
