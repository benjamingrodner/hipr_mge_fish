# Python script
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given cell segmentation and spot segmenation properties tables, assign spots to
cells
"""
# =============================================================================

import yaml
import sys
import pandas as pd
import numpy as np
import argparse

# =============================================================================


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-cfn', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-ch', '--spot_channel', dest ='spot_channel', type = str, help = '')
    parser.add_argument('-ss', '--spot_seg_fn', dest ='spot_seg_fn', type = str, help = '')
    parser.add_argument('-ssp', '--spot_seg_props_fn', dest ='spot_seg_props_fn', type = str, help = '')
    parser.add_argument('-mp', '--max_props_fn', dest ='max_props_fn', type = str, help = '')
    parser.add_argument('-fv', '--filt_vals_fn', dest ='filt_vals_fn', type = str, help = '')
    # parser.add_argument('-pp', '--pipeline_path', dest ='pipeline_path', type = str, help = '')
    parser.add_argument('-mpf', '--max_props_filt_fn', dest ='max_props_filt_fn', type = str, help = '')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load things
    spot_seg = np.load(args.spot_seg_fn)
    spot_props = pd.read_csv(args.spot_seg_props_fn)
    maxs = pd.read_csv(args.max_props_fn)
    filt_vals = pd.read_csv(args.filt_vals_fn)

    # filter spots based on intensity
    maxs_filt = maxs[maxs.intensity > filt_vals['int'].values[0]]
    # TODO: filter based on area


    # Save new spot properties file
    maxs_filt.to_csv(args.max_props_filt_fn)

    return

if __name__ == '__main__':
    main()
