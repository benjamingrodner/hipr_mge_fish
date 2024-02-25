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
from collections import defaultdict
import numba as nb
# import javabridge
# import bioformats
import aicspylibczi as aplc
import argparse



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
parser.add_argument('-cd', '--classif_dir', dest ='classif_dir', 
                    type = str, help = 'Directory to save pixel classification.')    
parser.add_argument('-mf', '--minmax_fn', dest ='minmax_fn', 
                    type = str, help = 'Filename to save min and max values table.')    
parser.add_argument('-r', '--raw_fns', dest ='raw_fns', 
                    type = str, nargs='+', help = 'Filename to save min and max values table.') 
parser.add_argument('-cid', '--classif_im_dir', dest ='classif_im_dir', 
                    type = str, help = 'Directory to save the classified images.')
parser.add_argument('-lf', '--legend_fn', dest ='legend_fn', 
                    type = str, help = 'Directory to save the classified images.')

# Parse the command-line arguments
args = parser.parse_args()

# Load config file
with open(args.config_fn, 'r') as f:
    config = yaml.safe_load(f)

# Special imports 
sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip
import fn_spectral_images as fsi


# =============================================================================
# Functions
# =============================================================================


def adjust_grayscale(stack, r_min, r_max, clip):
    im_sum = np.sum(stack, axis=2)
    r_sum_norm = (im_sum - r_min) / (r_max - r_min)
    r_clip = np.clip(r_sum_norm, clip[0], clip[1])
    return (r_clip - clip[0]) / (clip[1] - clip[0])


def load_cmap():
    col_df = pd.read_csv(config['color_map_fn'])
    colors = [eval(c) for c in col_df.color.values]
    return pd.Series(colors, index=col_df.sci_name).to_dict()


def classif_to_image(classifs, pix_ind, shape, plot_intensities, col_dict):
    im_clust = np.zeros(shape + (len(list(col_dict.values())[0]),))
    # im_clust = np.zeros(max.shape + (len(eval(barcode_color.color.values[0])),))
    for lab, x, y in zip(classifs, pix_ind[0], pix_ind[1]):
        col = col_dict[lab]
        # col = np.array(col_dict[lab]) * sum_norm[x,y]
        im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]
    return im_clust


def save_classif_image(im_clust, im_inches, m, mosaic_size, resolution):
    if m == (mosaic_size - 1):
        ip.plot_image(im_clust, im_inches=im_inches, scalebar_resolution=resolution)
    else:
        ip.plot_image(im_clust, im_inches=im_inches)
    out_bn = args.classif_im_dir + '/' + args.sn + '' + '_M_' + str(m) + '_pixelclassif'
    ip.save_png_pdf(out_bn)
    print('Saved:', out_bn + '.png')

def save_color_legend(labs_sort, col_dict):
    col_ordered = [col_dict[l] for l in labs_sort]
    fig, ax = ip.taxon_legend(
            taxon_names=labs_sort,
            taxon_colors=col_ordered
            )
    out_bn = os.path.splitext(args.legend_fn)[0]

    ip.save_png_pdf(out_bn)
    print('Saved:', args.legend_fn)



# =============================================================================
# Execution
# =============================================================================


def main():
    """
    Script to classify hiprfish images using a matrix based classifier.

    Usage:
        python sc_matrix_classify.py input_file output_file [--option1 OPTION1] [--option2 OPTION2]

    Example:
        python script_name.py data.csv processed_data.csv --option1 10 --option2 "value"
    """
    # initialize direcotries
    if not os.path.exists(args.classif_im_dir):
        os.makedirs(args.classif_im_dir)

    # Get filenames from directories 
    filenames = []
    for d in [args.shift_dir, args.classif_dir]:
        fns = glob.glob(d + '/*.*')
        filenames.append(sorted(fns))

    # Group shift filenames 
    shifts_fns = filenames[0]
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

    # Save classif
    dict_sciname_counts = defaultdict(lambda: 0)
    for i, sh_fns in enumerate(shifts_fns_group):
        plot_intensities = adjust_grayscale(
            stack=np.dstack([np.load(fn) for fn in sh_fns]), 
            r_min=vals.loc['min','sum'], 
            r_max=vals.loc['max','sum'], 
            clip=config['classif_im']['clip']
        )
        col_dict = load_cmap()

        # Load classif info 
        classif_df = pd.read_csv(filenames[1][i])
        im_classif = classif_to_image(
            classifs=classif_df.sci_name.values, 
            pix_ind=[classif_df[i].values for i in ['x','y']], 
            shape=plot_intensities.shape,
            plot_intensities=plot_intensities, 
            col_dict=col_dict
        )

        # Save classif image 
        save_classif_image(
            im_classif, 
            mosaic_size=len(shifts_fns_group), 
            m=i, 
            im_inches=config['classif_im']['im_inches'], 
            resolution=res_um_pix
        )

        # Pool classif info
        vcounts = classif_df.sci_name.value_counts()
        for k, v in vcounts.iteritems():
            dict_sciname_counts[k] += v
    
    # Plot the legend 
    scinames_sorted = sorted(dict_sciname_counts, key=dict_sciname_counts.get, reverse=True)
    save_color_legend(scinames_sorted, col_dict)



if __name__ == "__main__":
    main()
