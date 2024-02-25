
"""
Ben Grodner 2023
De Vlaminck Lab
Cornell University

Purpose: Used in pipeline for registering HiPRFISH images with MeGAFISH images
to correct x-y shifts. Upsamples HiPRFISH images so the resolution matches
MeGAFISH images, then uses phase cross correlation to overlay the two images.
"""

################################################################################
# Imports
################################################################################

import os
# import re
import sys
import yaml
import argparse
import numpy as np
# import pandas as pd
from cv2 import resize, INTER_CUBIC, INTER_NEAREST

################################################################################
# Functions
################################################################################
def center_image(im, dims, ul_corner):
    shp = im.shape
    if not all([dims[i] == shp[i] for i in range(len(dims))]):
        shp_new = dims if len(shp) == 2 else dims + (shp[2],)
        temp = np.zeros(shp_new)
        br_corner = np.array(ul_corner) + np.array(shp[:2])
        temp[ul_corner[0]:br_corner[0], ul_corner[1]:br_corner[1]] = im
        im = temp
    return im


def resize_hipr(in_fn, hipr_res, mega_res, dims='none', out_fn=False, ul_corner=(0,0)):
    im = np.load(in_fn)
    factor_resize = hipr_res / mega_res
    hipr_resize = resize(
            im,
            None,
            fx = factor_resize,
            fy = factor_resize,
            interpolation = INTER_NEAREST
            )
    if isinstance(dims, str): dims = hipr_resize.shape
    hipr_resize = center_image(hipr_resize, dims, ul_corner)
    if out_fn: np.save(out_fn, hipr_resize)
    return hipr_resize

################################################################################
# Script
################################################################################


def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-hm', '--hipr_max_fn', dest = 'hipr_max_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-hmr', '--hipr_max_resize_fn', dest = 'hipr_max_resize_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-hs', '--hipr_seg_fn', dest = 'hipr_seg_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-hsr', '--hipr_seg_resize_fn', dest = 'hipr_seg_resize_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-hpr', '--hipr_props_resize_fn', dest = 'hipr_props_resize_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-hsc', '--hipr_seg_col_fn', dest = 'hipr_seg_col_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-hscr', '--hipr_seg_col_resize_fn', dest = 'hipr_seg_col_resize_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-msv', '--mega_shift_vectors', dest = 'mega_shift_vectors', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-mr', '--mega_raw_fn', dest = 'mega_raw_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-mrs', '--mega_raw_shift_fn', dest = 'mega_raw_shift_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-mcs', '--mega_cell_seg_fns', dest = 'mega_cell_seg_fns', nargs = '+', default = [])
    parser.add_argument('-mcss', '--mega_cell_seg_shift_fns', dest = 'mega_cell_seg_shift_fns', nargs = '+', default = [])
    parser.add_argument('-mcps', '--mega_cell_props_shift_fns', dest = 'mega_cell_props_shift_fns', nargs = '+', default = [])
    parser.add_argument('-mss', '--mega_spot_seg_fns', dest = 'mega_spot_seg_fns', nargs = '+', default = [])
    parser.add_argument('-msss', '--mega_spot_seg_shift_fns', dest = 'mega_spot_seg_shift_fns', nargs = '+', default = [])
    parser.add_argument('-msps', '--mega_spot_props_shift_fns', dest = 'mega_spot_props_shift_fns', nargs = '+', default = [])

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    ### Setup
    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)
    with open(config['config_hipr'], 'r') as f:
        config_hipr = yaml.safe_load(f)
    with open(config['config_mega'], 'r') as f:
        config_mega = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config_mega['pipeline_path'] + '/' + config_mega['functions_path'])
    import fn_spectral_images as fsi
    import segmentation_func as sf

    # Define functions that use specialized modules
    def shift_mega(in_fns, out_fns):
        '''Globally define:  mega_shift_vector, max_shift, dims, ul_corner'''
        ims = []
        for in_fn, out_fn in zip(in_fns, out_fns):
            im = np.load(in_fn)
            if len(im.shape) == 2: im = im[...,None]
            im = center_image(im, dims, ul_corner)
            im_shift = fsi._shift_images([im], mega_shift_vector, max_shift=max_shift)
            np.save(out_fn, im_shift[0])
            ims.append(im_shift[0])
        return ims

    def get_mega_props(segs, type, out_fns):
        '''Globally define: raw_shift'''
        type='spot_seg'
        channels = config_mega[type]['channels']
        for seg, ch, out_fn in zip(segs, channels, out_fns):
            raw = raw_shift[:,:,ch]
            seg = seg.astype(np.int64)[:,:,0]
            props = sf.measure_regionprops(seg, raw=raw)
            props.to_csv(out_fn)


    ### Get Shift vectors for mega images
    # Get the Rescaled hiprfish image
    mega_res = config_mega['resolution']
    hipr_res = config_hipr['resolution']
    hipr_max_resize = resize_hipr(
            args.hipr_max_fn, hipr_res, mega_res
            )
    # Get mega cell image
    mega_raw = np.load(args.mega_raw_fn)
    reg_ch = config['mega_reg_ch_index']
    mega_cell = mega_raw[:,:,config_mega['cell_seg']['channels'][reg_ch]]
    # Which is the smaller image?
    mshp = mega_cell.shape
    hshp = hipr_max_resize.shape
    im_list = [mega_cell, hipr_max_resize]
    i_sml = np.argmin([mshp[0],hshp[0]])
    i_lrg = np.argmax([mshp[0],hshp[0]])
    sml = im_list[i_sml]
    lrg = im_list[i_lrg]
    # Get half the difference between sizes
    shp_dff = np.abs(np.array(hshp) - np.array(mshp)) // 2
    # Shift the smaller image so that it sits at the center of the larger image
    sml_shift = np.zeros(lrg.shape)
    corn_ind = np.array(shp_dff) + np.array(sml.shape)
    sml_shift[shp_dff[0]:corn_ind[0], shp_dff[1]:corn_ind[1]] = sml
    # reassign mega and hipr image var names
    im_shift_list = [0,0]
    im_shift_list[i_sml] = sml_shift
    im_shift_list[i_lrg] = lrg
    mega_shift = im_shift_list[0]
    hipr_shift = im_shift_list[1]
    # Get the shift vectors for the mega image
    image_list = [hipr_shift, mega_shift]
    shift_vectors = fsi._get_shift_vectors(image_list)
    np.save(args.mega_shift_vectors, shift_vectors)

    ### Shift the mega images
    # globally define vars for the shift function
    max_shift = config['max_shift']
    mega_shift_vector = [shift_vectors[1]]
    dims = lrg.shape
    ul_corner = shp_dff
    # run the shift function
    raw_shift = shift_mega([args.mega_raw_fn], [args.mega_raw_shift_fn])[0]
    cs_shifts = shift_mega(args.mega_cell_seg_fns, args.mega_cell_seg_shift_fns)
    ss_shifts = shift_mega(args.mega_spot_seg_fns, args.mega_spot_seg_shift_fns)
    # Run the props function
    get_mega_props(cs_shifts, 'cell_seg', args.mega_cell_props_shift_fns)
    get_mega_props(ss_shifts, 'spot_seg', args.mega_spot_props_shift_fns)

    ### Rescale the hiprfish images
    args.hipr_max_resize_fn
    _ = resize_hipr(
            args.hipr_max_fn, hipr_res, mega_res,
            out_fn=args.hipr_max_resize_fn,
            dims=dims, ul_corner=ul_corner
            )
    hipr_seg = resize_hipr(
            args.hipr_seg_fn, hipr_res, mega_res,
            out_fn=args.hipr_seg_resize_fn,
            dims=dims, ul_corner=ul_corner
            )
    _ = resize_hipr(
            args.hipr_seg_col_fn, hipr_res, mega_res,
            out_fn=args.hipr_seg_col_resize_fn,
            dims=dims, ul_corner=ul_corner
            )
    # Get hipr seg props after resizing
    hipr_seg = hipr_seg.astype(np.int64)
    hipr_props = sf.measure_regionprops(hipr_seg)
    hipr_props.to_csv(args.hipr_props_resize_fn)
    return

if __name__ == '__main__':
    main()






#####
