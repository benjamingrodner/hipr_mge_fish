# Python script
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given a segmented image, get the object properties table
"""
# =============================================================================

import yaml
import sys
import numpy as np
import pandas as pd
import argparse

# =============================================================================


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-cfn', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-st', '--seg_type', dest ='seg_type', type = str, help = '')
    parser.add_argument('-r', '--raw_fn', dest ='raw_fn', type = str, help = '')
    parser.add_argument('-s', '--seg_fn', dest ='seg_fn', type = str, help = '')
    parser.add_argument('-ch', '--channel', dest ='channel', type = str, default='all', help = '')
    parser.add_argument('-spfn', '--seg_props_fn', dest ='seg_props_fn', type = str, help = '')

    parser.add_argument('-m', '--maxima', dest ='maxima', type = str, default='', help = '')
    parser.add_argument('-lmfn', '--locmax_fn', dest ='locmax_fn', type = str, default='', help = '')
    parser.add_argument('-lmpfn', '--locmax_props_fn', dest ='locmax_props_fn', type = str, default='', help = '')
    parser.add_argument('-mtfn', '--multimax_table_fn', dest ='multimax_table_fn', type = str, default='', help = '')
    parser.add_argument('-sspfn', '--seg_split_props_fn', dest ='seg_split_props_fn', type = str, default='', help = '')

    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)
    pdict = config[args.seg_type]

    # Load segmentation and raw image
    raw = np.load(args.raw_fn)
    seg = np.load(args.seg_fn)

    # get 2d raw image
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    # sys.path.append(args.pipeline_path + '/functions')
    import segmentation_func as sf
    channels = pdict['channels'] if args.channel == 'all' else [int(args.channel)]
    raw_2D = sf.max_projection(raw, channels)
    # raw_2D = sf.max_projection(raw, pdict['channels'])

    # Get properties table
    props = sf.measure_regionprops(seg, raw_2D)
    # Save props table
    props.to_csv(args.seg_props_fn, index=False)

    # Find maxima, and slit segmentation when there are multiple local maxima
    if args.maxima or pdict['maxima']:
        import spot_funcs as spf
        # mask raw image by seg
        raw_masked = raw_2D * (seg > 0)
        # Get local maxima in spots, measure props, save both
        locmax = spf.peak_local_max(raw_masked,  indices=False,
                                    min_distance=config['local_max_mindist'])
        locmax = sf.label(locmax)
        print('HERE!  ',locmax.shape)
        np.save(args.locmax_fn, locmax)
        locmax_props = sf.regionprops_table(locmax, intensity_image = raw_2D,
                                            properties=['label','centroid','area',
                                                        'max_intensity'])
        locmax_props = pd.DataFrame(locmax_props)
        locmax_props.to_csv(args.locmax_props_fn, index=False)
        # get spots with multiple local maxima and save info
        multimax = spf.get_spots_with_multimax(seg, locmax)
        multimax.to_csv(args.multimax_table_fn, index=False)
        # Split the spots from the multimax table and save
        seg_split = spf.split_multimax_spots(multimax, props, seg, locmax_props,
                                             raw_2D)
        # Save the split properties
        seg_split_props = sf.measure_regionprops(seg_split, raw_2D)
        seg_split_props.to_csv(args.seg_split_props_fn, index=False)

    return

if __name__ == '__main__':
    main()
