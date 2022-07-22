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
    parser.add_argument('-cs', '--cell_seg_fn', dest ='cell_seg_fn', type = str, help = '')
    parser.add_argument('-mp', '--max_props_fn', dest ='max_props_fn', type = str, help = '')
    # parser.add_argument('-pp', '--pipeline_path', dest ='pipeline_path', type = str, help = '')
    parser.add_argument('-mpc', '--max_props_cid_fn', dest ='max_props_cid_fn', type = str, help = '')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load stuff
    maxs = pd.read_csv(args.max_props_fn)
    cell_seg = np.load(args.cell_seg_fn)

    # import custom stuff
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import spot_funcs as spf

    # assign spots to cells
    maxs_cid = spf.maxs_to_cell_id(maxs, cell_seg, config['max_dist_to_cell'])

    # Save new spot properties file
    maxs_cid.to_csv(args.max_props_cid_fn)

    return

if __name__ == '__main__':
    main()
