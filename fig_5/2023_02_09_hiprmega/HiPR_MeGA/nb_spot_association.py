# %% md

# # 2022_12_19_hiprmegafish

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
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from tqdm import tqdm
# import umap
# from cv2 import resize, INTER_CUBIC, INTER_NEAREST
from sklearn.neighbors import NearestNeighbors
from random import sample
from collections import defaultdict
from copy import copy
from sklearn.preprocessing import StandardScaler
# from umap import UMAP
from scipy.ndimage import gaussian_filter
# from hdbscan import HDBSCAN
from sklearn.cluster import KMeans


gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2023_02_09_hiprmega/HiPR_MeGA'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %%

# Go into your configuration file and adjust all of the 'Paths' so they are relative to the workdir.

# Also adjust the 'Inputs' parameters to fit the images.

# %% codecell
config_fn = 'config.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

config_fn = '../MeGAFISH/config_mgefish.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config_mega = yaml.safe_load(f)

config_hipr_fn = '../HiPRFISH/config_hipr.yaml' # relative path to config file from workdir

with open(config_hipr_fn, 'r') as f:
    config_hipr = yaml.safe_load(f)

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%reload_ext autoreload
%autoreload 2
sys.path.append(config_mega['pipeline_path'] + '/' + config_mega['functions_path'])
sys.path
import image_plots as ip
import fn_hiprfish_classifier as fhc
import fn_spectral_images as fsi
# import segmentation_func as sf
# import spot_funcs as spf
# import fn_manual_rois as fmr
# from fn_face_morpher import warp_image, weighted_average_points

# %% md

# Get sample names

# %% codecell
input_table_mge = pd.read_csv('../MeGAFISH/' + config_mega['input_table_fn'])
sample_names_mge = input_table_mge.sample_name.values
sample_names_mge

# %% codecell
input_table_hipr = pd.read_csv('../HiPRFISH/' + config_hipr['images']['image_list_table'])
sample_names_hipr = input_table_hipr.IMAGES.values
sample_names_hipr

# %% md

# =============================================================================
# ## Find the overlay
# =============================================================================

# Run the pipeline
