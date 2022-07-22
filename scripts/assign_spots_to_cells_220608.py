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
    parser.add_argument('-st', '--seg_type', dest ='seg_type', type = str, help = '')
    parser.add_argument('-r', '--raw_fn', dest ='raw_fn', type = str, help = '')
    parser.add_argument('-ss', '--spot_seg_fn', dest ='spot_seg_fn', type = str, help = '')
    parser.add_argument('-cs', '--cell_seg_fn', dest ='cell_seg_fn', type = str, help = '')
    parser.add_argument('-sp', '--spot_seg_props_fn', dest ='spot_seg_props_fn', type = str, help = '')
    parser.add_argument('-ch', '--channel', dest ='channel', type = str, default='all', help = '')

    parser.add_argument('-scid', '--spot_cid_fn', dest ='spot_cid_fn', type = str, help = '')
    parser.add_argument('-scm', '--spot_cid_multi_fn', dest ='spot_cid_multi_fn', type = str, help = '')
    parser.add_argument('-sspc', '--spot_seg_props_cid_fn', dest ='spot_seg_props_cid_fn', type = str, help = '')
    parser.add_argument('-ssc', '--spot_seg_cid_fn', dest ='spot_seg_cid_fn', type = str, help = '')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)
    pdict = config[args.seg_type]

    # Load stuff
    raw = np.load(args.raw_fn)
    spot_seg = np.load(args.spot_seg_fn)
    cell_seg = np.load(args.cell_seg_fn)
    spot_props = pd.read_csv(args.spot_seg_props_fn)

    # import custom stuff
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import spot_funcs as spf
    import segmentation_func as sf
    channels = pdict['channels'] if args.channel == 'all' else [int(args.channel)]
    raw_2D = sf.max_projection(raw, channels)

    # assign spots to cells
    spot_cid_df = spf.assign_spots_to_cells(spot_seg, cell_seg)
    spot_cid_df.to_csv(args.spot_cid_fn, index=False)

    # assign multimapped spots
    spot_cid_df_new = spf.assign_multimapped_spots(spot_cid_df, spot_props,
                                                  raw_2D, cell_seg)
    spot_cid_df_new.to_csv(args.spot_cid_multi_fn, index=False)

    # Merge assign with spot props
    spot_props = spot_props.merge(spot_cid_df_new, left_on='label',
                                  right_on='spot_id', how='left')
    spot_props['cell_id'] = spot_props['cell_id'].fillna(0)
    spot_props.to_csv(args.spot_seg_props_cid_fn)

    # Filter non-cell spots from the segmentaiton
    spot_seg_filt = spf.filter_seg_objects(spot_seg, spot_props, filter='cell_id')
    np.save(args.spot_seg_cid_fn, spot_seg_filt)

    return

if __name__ == '__main__':
    main()
