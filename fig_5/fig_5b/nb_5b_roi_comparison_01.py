# %% md

# # Figure 5b: AMR Plasmid HiPR MGE FISH

# # version 01:

# Used "hiprfish_imaging_py38" conda environment

# =============================================================================
# ## Setup
# =============================================================================

# Imports.

# %% codecell
import glob
import pandas as pd
import subprocess
import yaml
import gc
import os
import re
import javabridge
import bioformats
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import umap
from cv2 import resize, INTER_CUBIC, INTER_NEAREST

gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5b'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

# Go into your configuration file and adjust all of the 'Paths' so they are relative to the workdir.

# Also adjust the 'Inputs' parameters to fit the images.

# %% codecell
config_fn = 'config.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%reload_ext autoreload
%autoreload 2

sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
sys.path
import image_plots as ip
import segmentation_func as sf
import spot_funcs as spf
import fn_manual_rois as fmr

# %% md

# =============================================================================
# ## Get sister points between rois
# =============================================================================

# Get roi filenames and labels

# %% codecell
roi_dir = config['output_dir'] + '/roi_masks'
sn_hipr = '2022_03_19_plaquephagelytic_sample_bmg_probes_non_fov_tile1_round_2_mode_spec'
sn_mge = '20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_1_mode_airy_Airyscan_Processing_stitch'
sample_names = [sn_mge, sn_hipr]
roi_npy_fns = {}
roi_npy_labels = {}
for sn in sample_names:
    roi_sn_dir = roi_dir + '/' + sn
    # Load tiffs
    roi_tiff_fns = glob.glob(roi_sn_dir + '/*.tif')
    roi_tiff_fns.sort()
    roi_npy_fns[sn] = []
    roi_npy_labels[sn] = []
    for fn in roi_tiff_fns:
        roi_npy_fns[sn] += [fn]
        label = re.search(r'(?<=roi_)\d+', fn).group(0)
        roi_npy_labels[sn] += [label]

# %% md

# Extract cell channel ROIs and save as tiff for input to FIJI
