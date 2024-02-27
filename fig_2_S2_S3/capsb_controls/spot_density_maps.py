# %% md

# # Figure 3d: Density maps

# Run after running nb_3b_filtering.py. Use the "hiprfish_imaging_py38" conda environment.

# %% md
# ==============================================================================
# ## Setup
# ==============================================================================

# %% md

# Imports

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
from skimage.restoration import richardson_lucy
from scipy.stats import linregress
from sklearn.cluster import MiniBatchKMeans
from time import time
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

# %% md

# Move to the workdir

# %% codecell
# Absolute path
project_workdir = '/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3d'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

# Load all the variables from the segmentation pipeline

# %% codecell
config_fn = 'config.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% md

# Specify an output directory for plots

# %% codecell

if not os.path.exists(config['figure_dir']): os.makedirs(config['figure_dir'])

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%load_ext autoreload
%autoreload 2

sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip
import segmentation_func as sf
import spot_funcs as spf


# %% md

# Get sample names

# %% codecell
sample_names = pd.read_csv(config['input_table_fn']).sample_name.values
sample_names


# %% md

# ==============================================================================
# ## Plot spot density on map
# ==============================================================================

# %% codecell
# Get cell
c_ch = config['cell_seg']['channels'][0]
raw = ip.load_output_file(config, 'raw_fmt', sn)
cell = raw[:,:,c_ch]
cell_pix_values = np.sort(np.ravel(cell))
# %% codecell
# PLot intensities
thresh_cell=0.005
dims = (5,5)
s=1

vals = cell_pix_values[::200]
fig, ax = ip.general_plot(dims=dims)
ax.scatter(np.arange(vals.shape[0]), vals, s=s)
ax.plot([0,vals.shape[0]], [thresh_cell]*2, color='k')

# %% codecell
im_inches=5
clims=[(0,0.15),()]
cell_mask = cell > thresh_cell
pix_inds = np.where(cell_mask > 0)
pix_coords = np.array(pix_inds).T
ip.subplot_square_images([cell, cell_mask], (1,2), clims=clims, im_inches=im_inches)

# %% codecell
# Correct for edge effects
r_um = 20
r_pix = int(r_um / config['resolution'])




area_circle = np.pi * r_um**2
# Get circle
def get_circle_mask(dimx, dimy, center, radius):
    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    return (distance_from_center <= radius)
d = r_pix*2 + 1
circle_mask = get_circle_mask(d, d, (r_pix, r_pix), r_pix)
area_circle_pix = np.sum(circle_mask)
ip.plot_image(circle_mask)


# # %% codecell
# manually draw mask in fiji
edge_mask_fn = config['output_dir'] + '/edge_mask/' + sn + 'edgemask.tif'

# Get edge of mask
cell_mask_bound = ip.find_boundaries(cell_mask, mode='inner')
# Convolve with circle
cell_mask_edge = fftconvolve(cell_mask_bound, circle_mask)
# multiply by mask
cell_mask_edge *= np.pad(cell_mask, np.int(r_pix), mode='edge')
cell_mask_edge =

# plot
ip.plot_image(cell_mask_edge > 0)


# %% codecell
# For each pixel coordinate
thresh_area_frac = 0.9

thresh_circle_area = area_circle_pix * thresh_area_frac
cell_mask_pad = np.pad(cell_mask, np.int(r_pix), mode='edge')
area_list = []
um_sq_per_pix = config['resolution']**2
for i,j in tqdm(pix_coords):
    # Get bbox
    i += r_pix
    j += r_pix
    bbox = cell_mask_pad[i-r_pix: i+r_pix+1, j-r_pix: j+r_pix+1]
    if np.sum(bbox) < thresh_circle_area:
        # multiply bbox by circle
        reduced_mask = circle_mask * bbox
        area = np.sum(reduced_mask) * um_sq_per_pix
    else:
        area = area_circle
    # add to area list
    area_list.append(area)

# %% codecell
# plot edge correction
cell_edge_map = np.zeros(cell.shape)
cell_edge_map[pix_inds] = area_list
fig, ax, cbar = ip.plot_image(cell_edge_map)
background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
background = np.dstack(background)
ax.imshow(background)

# %% codecell
# Calculate spot densities at subset of pixels
subset=2
chan_densities = []
spot_coords_sub = np.array(spot_coords)[::subset]
area_list_sub = np.array(area_list)[::subset]
for s_ch in config['spot_seg']['channels']:
    print(s_ch)
    # Get spot coords
    props = ip.load_output_file(config, 'spot_props_bg_int_fmt', sn, spot_chan=s_ch)
    bools = (props.area_thresh.values * props.int_thresh.values) == 1
    props_filt = props[bools]
    spot_coords = [list(eval(c)) for c in props_filt.centroid.values]
    print(len(spot_coords))
    # Generate knn object with radius
    nbrs = NearestNeighbors(radius=r_pix).fit(spot_coords)
    # Get radius knn for each pixel
    t0 = time()
    print(t0)
    dists, inds = nbrs.radius_neighbors(pix_coords)
    t1 = time()
    print(t1)
    print(t1-t0)
    # Apply density value to cell pixels
    pix_counts = [i.shape[0] for i in inds]
    pix_density = np.array(pix_counts) / np.array(area_list)
    chan_densities.append(pix_density)


# %% codecell
# Plot
clims = (0, np.max(chan_densities))
for s_ch, pix_density in zip(config['spot_seg']['channels'][:1], chan_densities):
    print(s_ch)
    cell_spot_density = np.zeros(cell.shape)
    cell_spot_density[pix_inds] = pix_density
    fig, ax, cbar = ip.plot_image(
            cell_spot_density,
            cmap='cividis',
            scalebar_resolution=config['resolution'],
            clims=clims,
            cbar_ori='vertical'
            )
    background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
    background = np.dstack(background)
    ax.imshow(background)
    # plt.figure(fig)
    # out_dir = config['output_dir'] + '/density_maps'
    # if not os.path.exists(out_dir): os.makedirs(out_dir)
    # output_bn = out_dir + '/' + sn + '_spot_density_map_chan_' + str(s_ch)
    # ip.save_png_pdf(output_bn)
    # plt.figure(cbar)
    out_dir = config['output_dir'] + '/density_maps'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    output_bn = out_dir + '/' + sn + '_spot_density_map_chan_' + str(s_ch) + '_cbar'
    ip.save_png_pdf(output_bn)
    plt.show()
    plt.close()
