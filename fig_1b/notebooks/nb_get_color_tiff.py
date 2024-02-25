# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_09
# =============================================================================
"""
The notebook Has Been Built for...generating figures for the raw images from
the GFP E. coli experiment

For use with the hiprfish_imaging_py38 conda env
"""
# %% codecell
# =============================================================================
# Setup
# =============================================================================
# Modify
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/mobile_elements/experiments/2022_01_22_gelgfp'
                    # Absolute path to the project work directory
image_analysis_code_path = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code'
tif_dir = '/fs/cbsuvlaminck2/workdir/Data/bmg224/2022/brc_imaging/2022_01_22_gelgfp/figpngs'
ext = '_c1-2.tif'
seg_dir = 'segmentation'
fig_dir = 'figures'

# %% codecell
# Imports
import sys
import os
import re
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# special modules
sys.path.append(image_analysis_code_path)
import image_functions as imfn
import image_plots as ip
# %% codecell
# =============================================================================
# Load tiffs and zoom regions
# =============================================================================
# get sample names
sample_names = imfn.get_sample_names(data_dir=tif_dir, ext=ext)
sample_names.sort()
sample_names
# %% codecell
# remove spaces
sample_names_ = [re.sub('\s','_',sn) for sn in sample_names]
sample_names_ = [re.sub('c.1','c1',sn) for sn in sample_names_]
sample_names_
# %% codecell
# set up sample_name zoom dict
zoom_dict = {}

# %% codecell
# load image
i = 11
zoom_dims=(350,350)
sample_name = sample_names[i]
sample_name_ = sample_names_[i]
fn = data_dir + '/' + sample_name + ext
print(fn)
raw_filename = seg_dir + '/raw_npy/' + sample_name_ + '.npy'
image = np.load(raw_filename)
im_spot = image[:,:,1]
im_cell = image[:,:,0]

# %% codecell
# Show full image
plt.figure(figsize=(20,20))
plt.imshow(im_cell, cmap='inferno')

# %% codecell
# Pick zoom regions by hand
tr_corner = [2300, 3000]
zc = [tr_corner[0],tr_corner[0]+zoom_dims[0],tr_corner[1],tr_corner[1]+zoom_dims[1]]
fig, ax = ip.plot_rgb(fn, zoom_coords=zc)
out_basename = fig_dir + '/' + sample_name_ + '_rgb_tiff_zoom'
ip.save_png_pdf(out_basename, bbox_inches=False)

# %% codecell
# # find random zoom regions for new images
# zm, zc = ip.get_zoom_region(im, dims=zoom_dims)
# fig, ax = ip.plot_rgb(fn, zoom_coords=zc)
# out_basename = fig_dir + '/' + sample_name_ + '_rgb_tiff_zoom'
# ip.save_png_pdf(out_basename, bbox_inches=False)

# %% codecell
# assign the zoom region to the sample name
zoom_dict[sample_name_] = [zc, zoom_dims]
len(zoom_dict)

# %% codecell
# Export zoom coords table
zoom_coords_df = pd.DataFrame(sample_names_, columns=['sample_name'])
zoom_coords_df['coords'] = [zoom_dict[sn][0] for sn in sample_names_]
zoom_coords_df
# %% codecell
zc_output_fn = fig_dir + '/zoom_coords_new.csv'
zoom_coords_df.to_csv(zc_output_fn)
