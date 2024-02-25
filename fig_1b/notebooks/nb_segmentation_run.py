# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_28
# =============================================================================
"""
The notebook Has Been Built for...Running the segmentation_pipeline

For use with the 'hiprfish_imaging_py38' conda env
"""
# %% codecell
# =============================================================================
# Setup
# =============================================================================
# Modify
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/mobile_elements/experiments/2022_01_22_gelgfp'
                    # Absolute path to the project work directory
config_fn = 'config.yaml' # relative path to config file from workdir

# %% codecell
# Imports
import glob
import pandas as pd
import subprocess
import yaml
import gc
import os
import javabridge
import bioformats
import sys

# %% codecell
# Set up notebook stuff
%load_ext autoreload
%autoreload 2
gc.enable()

# %% codecell
# Move to notebook directory
os.chdir(project_workdir)
os.getcwd()

# %% codecell
# load config file
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% codecell
# Function imports
sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip
import segmentation_func as sf
import spot_funcs as spf

# %% codecell
# =============================================================================
# Get test image
# =============================================================================
#  get input files automatically
input_filenames = glob.glob(config['input_dir'] + '/*' + config['input_ext'])
input_filenames.sort()
print(len(input_filenames))
input_filenames

# %% codecell
# Select one image to set parameters
test_index = 11
input_fn = input_filenames[test_index]
input_fn

# %% codecell
# load the image
javabridge.start_vm(class_path=bioformats.JARS)
input = bioformats.load_image(input_fn)
input.shape

# %% codecell
# show the image
clims = [(),(0,0.05)]
n_channels = input.shape[2]
im_list = [input[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% codecell
# subset the image
sr = [(300,500),(250,450)]
input_sub = input[sr[0][0]:sr[0][1],sr[1][0]:sr[1][1]]
im_list = [input_sub[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% codecell
# =============================================================================
# Set cell segmetnation parameters
# =============================================================================
# get cell channel
im_cell_list = [input_sub[:,:,i] for i in config['cell_seg']['channels']]
if len(im_cell_list) > 1:
    im_cell = np.max((np.dstack(im_cell_list)), axis=2)
else:
    im_cell = im_cell_list[0]

# %% codecell
# set cell segmentation parameters in config file
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['cell_seg']

# %% codecell
# run segmentation
im_cell_pre = sf.pre_process(
    im_cell,
    log=pdict['pre_log'],
    denoise=pdict['pre_denoise'],
    gauss=pdict['pre_gauss']
    )
im_cell_mask = sf.get_background_mask(
    im_cell,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold']
    )
im_cell_seg = sf.segment(
    im_cell_pre,
    background_mask = im_cell_mask,
    n_clust=pdict['n_clust'],
    small_objects=pdict['small_objects']
    )
im_cell_seg.shape

# %% codecell
# show seg process
seg_rgb = ip.seg2rgb(im_cell_seg)
im_list = [im_cell, im_cell_pre, im_cell_mask, seg_rgb]
ip.subplot_square_images(im_list, (1,4))

# %% codecell
# =============================================================================
# Set spot segmetnation parameters
# =============================================================================
# get cell channel
im_spot_list = [input_sub[:,:,i] for i in config['spot_seg']['channels']]
if len(im_spot_list) > 1:
    im_spot = np.max((np.dstack(im_spot_list)), axis=2)
else:
    im_spot = im_spot_list[0]

# %% codecell
# set cell segmentation parameters in config file
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['spot_seg']

# %% codecell
# run segmentation
im_spot_pre = sf.pre_process(
    im_spot,
    log=pdict['pre_log'],
    denoise=pdict['pre_denoise'],
    gauss=pdict['pre_gauss']
    )
im_spot_mask = sf.get_background_mask(
    im_spot,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold']
    )
im_spot_seg = sf.segment(
    im_spot_pre,
    background_mask = im_spot_mask,
    n_clust=pdict['n_clust'],
    small_objects=pdict['small_objects']
    )
im_spot_seg.shape

# %% codecell
# show seg process
seg_rgb = ip.seg2rgb(im_spot_seg)
im_list = [im_spot, im_spot_pre, im_spot_mask, seg_rgb]
ip.subplot_square_images(im_list, (1,4))


# %% codecell
# =============================================================================
# Set spot max parameters
# =============================================================================
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# maxs = spf.peak_local_max(im_spot, min_distance = config['local_max_mindist'])
ma = spf._get_merged_peaks(im_spot, min_distance=config['local_max_mindist'])
# is_peak = spf.peak_local_max(im_spot, indices=False, min_distance=config['local_max_mindist']) # outputs bool image
# is_peak.shape
# labels = spf.label(is_peak)[0]
# merged_peaks = spf.center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
# ma = np.array(merged_peaks)
fig, ax, cbar = ip.plot_image(im_spot, cmap='inferno', im_inches=20)
ax.scatter(ma[:,1],ma[:,0], s=50, color=(0,1,0))
ax = ip.plot_seg_outline(ax, im_spot_seg, col=(0,0.8,0.8))
# ax.set_xlim((300,400))
# ax.set_ylim((425,475))

# %% codecell
# =============================================================================
# Run the pipeline
# =============================================================================

# %% codecell
# Write the test input_table
input_fns_split = [os.path.split(fn)[1] for fn in [input_fn]]
sample_names = [os.path.splitext(fn)[0] for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['input_table_fn'], index=False)
input_table


# %% codecell
# Write the full input_table
input_fns_split = [os.path.split(fn)[1] for fn in input_filenames]
sample_names = [os.path.splitext(fn)[0] for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['input_table_fn'], index=False)
input_table

# %% codecell
# Execute the snakemake
dry_run = False  # Just create DAG if True
n_cores = 1  # number of allowed cores for the snakemake to use
force_run = 'get_spot_seg_props'  # False if none

snakefile = config['pipeline_path'] + '/' + config['snakefile']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
command = [
    'snakemake', '-s', snakefile, '--configfile', config_fn, '-j', str(n_cores),
    dr, fr
    ]
print(" ".join(command))
subprocess.check_call(command)



# %% clims_cell


import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8],[1,2,1,2]]).T
b = list(a)
b

c = np.unique(a[:,2])
c


d = pd.DataFrame([[x,[(y[0],y[1]) for y in list(a) if y[2]==x]] for x in c])
d

from itertools import groupby
from operator import itemgetter
b.sort(key=itemgetter(2))
b
e = groupby(b, itemgetter(2))
e
[[i[0],[j for j in i[1]]] for i in e]

a = pd.DataFrame(np.array([[1,2],[1,2]]))
a


ma_df = pd.DataFrame(ma, columns=['row','col'])
maxs_cid = spf.maxs_to_cell_id(ma_df, im_cell_seg, search_radius=5)
maxs_cid
