
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
import pandas as pd
from cv2 import resize, INTER_CUBIC, INTER_NEAREST

################################################################################
# Functions
################################################################################


################################################################################
# Script
################################################################################


def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-msps', '--mega_spot_props_shift_fns', dest = 'mega_spot_props_shift_fns', nargs = '+', default = [])
    parser.add_argument('-mrs', '--mega_raw_shift_fn', dest = 'mega_raw_shift_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-om', '--overlay_mega_fns', dest = 'overlay_mega_fns', nargs = '+', default = [])
    # parser.add_argument('-hmr', '--hipr_max_resize_fn', dest = 'hipr_max_resize_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')

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
    sys.path.append(config['pipeline_path'] + '/' + config_mega['functions_path'])
    import image_plots as ip
    import segmentation_func as sf

    # Plot spots on Airy
    d = config['overlay_mega']
    mega_raw = np.load(args.mega_raw_shift_fn)
    mega_cell = mega_raw[:,:,d['raw_chan']]
    for props_fn, out_fn in zip(
            args.mega_spot_props_shift_fns, args.overlay_mega_fns):
        # Load hiprfish image
        # Load spot seg properties
        spot_props_shift = pd.read_csv(props_fn)
        # Plot col
        mge_res = config_mega['resolution']
        fig, ax, cbar = ip.plot_image(
                mega_cell,
                clims=eval(d['clims']),
                cmap=d['cmap'],
                scalebar_resolution=mge_res,
                im_inches=d['im_inches']
                )
        # Plot spots
        if spot_props_shift.shape[0] > 0:
            ref_pts = [eval(c) for c in spot_props_shift['centroid'].values]
            ref_pts = np.rint(ref_pts).astype(np.int64)
            ref_pts_arr = np.array(ref_pts)
            spot_int = spot_props_shift.max_intensity.values
            spot_int /= np.max(spot_int)
            spot_int[spot_int > d['ceil']] = d['ceil']
            spot_int /= d['ceil']
            marker_size_arr = d['marker_size'] * spot_int
            ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
                        marker=d['marker'], s=marker_size_arr, color=d['spot_col'],
                        linewidths=d['linewidths'], edgecolors='none'
                        )
        # Save plot
        ip.plt.sca(ax)
        out_bn = os.path.splitext(out_fn)[0]
        ip.save_png_pdf(out_bn, dpi=d['dpi'])

    return

if __name__ == '__main__':
    main()






#####
