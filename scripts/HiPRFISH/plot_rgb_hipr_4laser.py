
"""
Ben Grodner 2023
De Vlaminck Lab
Cornell University

HiPRFISH processing pipeline script

Plot several channels from the spectral image as an RGB image
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd



def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    # parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-rf', '--reg_fn', dest = 'reg_fn', type = str, default = '', help = 'Input classified spectra table')
    parser.add_argument('-of', '--out_fn', dest = 'out_fn', type = str, default = '', help = 'Input classified spectra table')

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi
    import image_plots as ip

    stack = np.load(args.reg_fn)

    chans = config['rgb_channels']
    cols = config['rgb_colors']
    # ulims = config['rgb_ulims']

    max_len = config['remove_405']['max_len']

    if stack.shape[2] > max_len:
        stack = stack[:,:,max_len:]
        
    lchan = config['laser_channels']
    raws_las = [stack[:,:,lchan[i]:lchan[i+1]] for i in range(len(lchan)-1)]
    if chans == 'max_project':
        # Max projection for each laser
        raws_chan = [np.max(r, axis=2) for r in raws_las]
    elif chans == 'sum_project':
        # sum projection for each laser
        raws_chan = [np.sum(r, axis=2) for r in raws_las]
    else:
        # Only a few channels
        raws_chan = [stack[:,:,ch] for ch in chans]

    # raws_chan = [stack[:,:,c] for c in chans]
    # Adjust intensities
    # raws_chan_norm = [im / np.max(im) for im in raws_chan]
    # raws_chan_norm_adj = []
    # for im, ul in zip(raws_chan_norm, ulims):
    #     im[im>ul] = ul
    #     raws_chan_norm_adj.append(im/ul)
    clips = config['rgb_clips']
    raws_chan_clip = [np.clip(im,c[0],c[1]) for im, c in zip(raws_chan, clips)]
    raws_chan_norm_adj = [
        (im - c[0]) / (c[1] - c[0]) for im, c in zip(raws_chan_clip, clips)
    ]

    raws_chan_col = []
    for c, im in zip(cols, raws_chan_norm_adj):
        raws_chan_col.append(im[...,None] * np.array(c)[None,None,:])
    # raws_chan_col = [im[...,None] * np.array(c)[None,None,:] for c, im in zip(cols, raws_chan_norm_adj)]
    chan_rgb = np.zeros(stack.shape[:2] + (len(cols[0]),))
    for im in raws_chan_col:
        chan_rgb += im
    ip.plot_image(chan_rgb, im_inches=config['rgb_im_inches'])
    out_bn = os.path.splitext(args.out_fn)[0]
    ip.save_png_pdf(out_bn)

    return

if __name__ == '__main__':
    main()






#####
