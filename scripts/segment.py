# Python script
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_28
# =============================================================================
"""
Segment an image using the neighbor2d algorithm
"""
# =============================================================================

import yaml
import sys
import numpy as np
import argparse

# =============================================================================


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('in_fn', type = str, help = 'Input filename')
    parser.add_argument('out_fn', type = str, help = 'Output filename')
    parser.add_argument('-cfn', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-st', '--seg_type', dest ='seg_type', type = str, help = '')
    # parser.add_argument('-pp', '--pipeline_path', dest ='pipeline_path', type = str, help = '')
    parser.add_argument('-pfn', '--process_fn', dest ='process_fn', type = str, help = '')
    parser.add_argument('-ch', '--channel', dest ='channel', type = str, default='all', help = '')
    args = parser.parse_args()

    # set cell segmentation parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)
    pdict = config[args.seg_type]

    # Load image
    im_full = np.load(args.in_fn)

    # get 2d raw image
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import segmentation_func as sf
    channels = pdict['channels'] if args.channel == 'all' else [int(args.channel)]
    im = sf.max_projection(im_full, channels)
    # for ch in pdict['channels']:
        # im = im_full[:,:,ch]
    # run segmentation
    im_pre = sf.pre_process(
        im,
        log=pdict['pre_log'],
        denoise=pdict['pre_denoise'],
        gauss=pdict['pre_gauss'],
        diff_gauss=eval(pdict['diff_gauss'])
        )
    im_mask = sf.get_background_mask(
        im,
        bg_filter=pdict['bg_filter'],
        bg_log=pdict['bg_log'],
        bg_smoothing=pdict['bg_smoothing'],
        n_clust_bg=pdict['n_clust_bg'],
        top_n_clust_bg=pdict['top_n_clust_bg'],
        bg_threshold=pdict['bg_threshold']
        )
    im_seg = sf.segment(
        im_pre,
        background_mask = im_mask,
        n_clust=pdict['n_clust'],
        small_objects=pdict['small_objects']
        )



    # save segmentation
    np.save(args.out_fn, im_seg)

    # Save image of processing steps
    import image_plots as ip
    seg_rgb = ip.seg2rgb(im_seg)
    im_list = [im, im_pre, im_mask, seg_rgb]
    ip.subplot_square_images(im_list, (1,4))
    ip.save_fig(args.process_fn, dpi=config['seg_process_dpi'])
    return

if __name__ == '__main__':
    main()
