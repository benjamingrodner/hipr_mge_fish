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
    parser.add_argument('-cp', '--cell_seg_props_fn', dest ='cell_seg_props_fn', type = str, help = '')
    parser.add_argument('-cpm', '--cell_props_spot_max_fn', dest ='cell_props_spot_max_fn', type = str, help = '')
    parser.add_argument('-rf', '--raw_fn', dest ='raw_fn', type = str, help = '')
    parser.add_argument('-ch', '--channel_spot', dest ='channel_spot', type = str, help = '')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # import custom modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import segmentation_func as sf
    import spot_funcs as spf

    # Load files
    cell_seg = np.load(args.cell_seg_fn)
    cell_seg_props = pd.read_csv(args.cell_seg_props_fn)
    raw = np.load(args.raw_fn)
    raw_2D = sf.max_projection(raw, [int(args.channel_spot)])

    # Get max spot value by cell seg
    spot_max_int = spf.get_cell_spot_maxs(cell_seg_props, cell_seg, raw_2D,
                                         r=config['max_dist_to_cell'])

    # Save new cell seg properties
    cell_seg_props['cell_spot_max_int'] = spot_max_int
    cell_seg_props.to_csv(args.cell_props_spot_max_fn)

    return

if __name__ == '__main__':
    main()
