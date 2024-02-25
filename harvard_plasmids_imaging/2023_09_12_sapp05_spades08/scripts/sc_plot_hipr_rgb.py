# =============================================================================
# Load arguments, configuration file, and special imports
# =============================================================================

import glob
import sys
import os
# import gc
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import yaml
# import re
# import joblib
# from sklearn.cluster import AgglomerativeClustering
# from tqdm import tqdm
# from scipy.ndimage import gaussian_filter
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# from collections import defaultdict
# import numba as nb
# import javabridge
# import bioformats
import aicspylibczi as aplc
import argparse
from datetime import datetime



# =============================================================================
# Load arguments, configuration file, and special imports
# =============================================================================

parser = argparse.ArgumentParser(description='Script Description')

# Add arguments to the parser

parser.add_argument('-cfn', '--config_fn', dest ='config_fn', 
                    type = str, help = 'Path to configureaiton file')
parser.add_argument('-sn', '--sample_name', dest ='sn', 
                    type = str, help = 'Sample name wildcard')
parser.add_argument('-sd', '--shift_dir', dest ='shift_dir', 
                    type = str, help = 'Directory to save channel shifts.') 
parser.add_argument('-mf', '--minmax_fn', dest ='minmax_fn', 
                    type = str, help = 'Filename to save min and max values table.')    
parser.add_argument('-r', '--raw_fns', dest ='raw_fns', 
                    type = str, nargs='+', help = 'Filename to save min and max values table.') 
parser.add_argument('-rd', '--rgb_dir', dest ='rgb_dir', 
                    type = str, help = 'Directory to save the RGB images.')
parser.add_argument('-rdf', '--rgb_done_fn', dest ='rgb_done_fn', 
                    type = str, help = 'Snakemake file.')

# Parse the command-line arguments
args = parser.parse_args()

# Load config file
with open(args.config_fn, 'r') as f:
    config = yaml.safe_load(f)

# Special imports 
sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip


# =============================================================================
# Functions
# =============================================================================

def get_rgb(m_raws_shift, r_mins, r_maxs, clips):
    rgb = []
    for i, r in enumerate(m_raws_shift):
        r_sum = np.sum(r, axis=2)
        r_sum_norm = (r_sum - r_mins[i]) / (r_maxs[i] - r_mins[i])
        r_clip = np.clip(r_sum_norm, clips[i][0], clips[i][1])
        r_clip_norm = (r_clip - clips[i][0]) / (clips[i][1] - clips[i][0])
        rgb.append(r_clip_norm)
    return np.dstack(rgb)

        
def save_rgb(rgb, mosaic_size, m, im_inches, resolution):
        if m == (mosaic_size-1):
            ip.plot_image(rgb, im_inches=im_inches, scalebar_resolution=resolution)
        else:
            ip.plot_image(rgb, im_inches=im_inches)
        out_bn = args.rgb_dir + '/' + args.sn + '' + '_M_' + str(m) + '_rgb'
        ip.save_png_pdf(out_bn)
        print('Saved:', out_bn)  



# =============================================================================
# Execution
# =============================================================================


def main():
    """
    Script to plot RGB images based on sum projection of hiprfish 3 laser imaging.

    Used in Snakemake pipeline 

    """
    # initialize direcotries
    if not os.path.exists(args.rgb_dir):
        os.makedirs(args.rgb_dir)

    # Get filenames from directories 
    shifts_fns = glob.glob(args.shift_dir + '/*.*')
    n = len(config['lasers'])
    shifts_fns_group = [shifts_fns[i:i+n] for i in range(0, (len(shifts_fns) - n + 1), n)]

    # Get max values 
    vals = pd.read_csv(args.minmax_fn, index_col=0)

    # Get resolution
    czi = aplc.CziFile(args.raw_fns[0])
    for n in czi.meta.iter():
        if 'Scaling' in n.tag:
            if 'X' in n.tag:
                resolution = float(n.text)
    res_um_pix = resolution * 10**6

    # Save rgbs
    for i, sh_fns in enumerate(shifts_fns_group):
        shifts = [np.load(fn) for fn in sh_fns]
        r_mins = vals.loc['min',:].values
        r_maxs = vals.loc['max',:].values
        clips = config['rgb']['clips']
        rgb = get_rgb(shifts, r_mins=r_mins, r_maxs=r_maxs, clips=clips)
        save_rgb(
            rgb, 
            mosaic_size=len(shifts_fns_group), 
            m=i, 
            im_inches=config['rgb']['im_inches'], 
            resolution=res_um_pix
        )

    # write snakemake file 
    with open(args.rgb_done_fn, 'w') as f:
        f.write('Done:' + str(datetime.now()))


if __name__ == "__main__":
    main()
