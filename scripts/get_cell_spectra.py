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
    parser.add_argument('-r', '--raw_fn', dest ='raw_fn', type = str, help = '')
    parser.add_argument('-s', '--seg_fn', dest ='seg_fn', type = str, help = '')
    # parser.add_argument('-st', '--seg_type', dest ='seg_type', type = str, help = '')
    parser.add_argument('-sp', '--seg_props_fn', dest ='seg_props_fn', type = str, help = '')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)
    # pdict = config[args.seg_type]

    # Load segmentation and raw image
    raw = np.load(args.raw_fn)
    seg = np.load(args.seg_fn)

    # get 2d raw image
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    # sys.path.append(args.pipeline_path + '/functions')
    import segmentation_func as sf
    import fn_spectral_images as fsi
    # channels = pdict['channels'] if args.channel == 'all' else [int(args.channel)]
    # raw_2D = sf.max_projection(raw, channels)
    # raw_2D = sf.max_projection(raw, pdict['channels'])

    # Get properties table
    props = sf.measure_regionprops(seg)
    avgint = fsi.get_cell_average_spectra(seg, raw)
    props = props.join(pd.DataFrame(avgint))

    # Save props table
    props.to_csv(args.seg_props_fn, index=False)

    return

if __name__ == '__main__':
    main()
