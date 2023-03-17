# %% md

# # Design an alternate classifier for hiprfish spectra

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
# import javabridge
# import bioformats
import sys
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from PIL import Image
from tqdm import tqdm
import umap
from cv2 import resize, INTER_CUBIC, INTER_NEAREST
from sklearn.neighbors import NearestNeighbors
from random import sample
from collections import defaultdict
import joblib
# from copy import copy

gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2022_12_19_hiprmegafish/HiPRFISH'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %%

# Go into your configuration file and adjust all of the 'Paths' so they are relative to the workdir.

# Also adjust the 'Inputs' parameters to fit the images.

# %% codecell
config_hipr_fn = '../HiPRFISH/config_hipr.yaml' # relative path to config file from workdir

with open(config_hipr_fn, 'r') as f:
    config_hipr = yaml.safe_load(f)

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%reload_ext autoreload
%autoreload 2
sys.path.append(config_hipr['pipeline_path'] + '/' + config_mega['functions_path'])
sys.path
import image_plots as ip
import segmentation_func as sf
# import spot_funcs as spf
# import fn_manual_rois as fmr
# from fn_face_morpher import warp_image, weighted_average_points

# %% md

# Get sample names

# %% codecell
input_table_hipr = pd.read_csv('../HiPRFISH/' + config_hipr['images']['image_list_table'])
sample_names = input_table_hipr.IMAGES.values
sample_names

# %% md

# =============================================================================
# ## Get the training data
# =============================================================================

# %% codecell

test_dir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_06_restain/HiPRFISH/reference_training/test'
training_data = pd.read_csv(test_dir + '/NSIMS_2000_PROBEDESIGN_welch2016_5b_no_633_channel_FUNCTION_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_5b_OBJ_umap_transform.csv')
scaler_full = joblib.load(test_dir + '/NSIMS_2000_PROBEDESIGN_welch2016_5b_no_633_channel_FUNCTION_interaction_simulated_excitation_adjusted_normalized_umap_transformed_biofilm_5b_OBJ_scaler.pkl')

# %% codecell
dir(scaler_full)
