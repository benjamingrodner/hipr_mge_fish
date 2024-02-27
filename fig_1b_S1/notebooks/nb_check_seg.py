# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_25
# =============================================================================
"""
The notebook Has Been Built for...checking segmentations

For use with the hiprfish_imaging conda env
"""
# %% codecell
# =============================================================================
# Setup
# =============================================================================
image_analysis_code_dir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code'
nb_dir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/mobile_elements/experiments/2022_01_22_gelgfp/notebooks'
data_dir = '../data/test'
fig_dir = '../figures'
image_processing_dir = '../image_processing'
sample_names = [
        '2022_01_22_gelgfp_pk_pos_gfp_pos_fov_tile6_Airyscan_Processing_shad_stitch'
        ]
exts = ['_Airyscan Processing_shad_stitch.czi']*2
channel_names = ['smeS','MefE','mexZ','16s rRNA']

# %% codecell
# Imports
import sys
sys.path.append(image_analysis_code_dir)
import image_plots as ip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import os

# %% codecell
# Set up notebook stuff
%load_ext autoreload
%autoreload 2
gc.enable()

# %% codecell
# Move to notebook directory
os.chdir(nb_dir)
os.getcwd()

# %% codecell
# Load cell numpys
sn = sample_names[0]
raw = np.load(image_processing_dir + '/' + sn + '_cell_seg_split/row_0_col_0.npy')
seg = np.load(image_processing_dir + '/' + sn + '_cell_seg_split/row_0_col_0_seg.npy')
from skimage.color import label2rgb
seg_rgb = label2rgb(seg,  bg_label = 0, bg_color = (0,0,0))
seg.shape

# %% codecell
# show seg
ip.subplot_square_images([raw, seg_rgb], (1,2), cmaps=['inferno',''])
# %% codecell
# Load spot numpys
sn = sample_names[0]
raw = np.load(image_processing_dir + '/' + sn + '_spot_seg_split/row_0_col_0.npy')
seg = np.load(image_processing_dir + '/' + sn + '_spot_seg_split/row_0_col_0_seg.npy')
from skimage.color import label2rgb
seg_rgb = label2rgb(seg,  bg_label = 0, bg_color = (0,0,0))
seg.shape

# %% codecell
# show seg
ip.subplot_square_images([raw, seg_rgb], (1,2), cmaps=['inferno',''])
