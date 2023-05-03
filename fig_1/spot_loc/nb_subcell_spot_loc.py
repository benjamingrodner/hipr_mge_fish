# %% md

# # HiPRFISH processing

# Used the "hiprfish_imaging_py38" conda environment.

# Takes as input Spectral images from i880 confocal with hiprfish spectral barcoding

# ==============================================================================
# ## Setup
# ==============================================================================

# %% md

# Imports

# %% codecell
import glob
import sys
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import re
from collections import defaultdict
import javabridge
import bioformats
from tqdm import tqdm
import aicspylibczi as aplc




# %% md

# Move to the workdir

# %% codecell
# Absolute path
project_workdir = '/workdir/bmg224/manuscripts/mgefish/code/fig_1/spot_loc'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory


# %% md

# Define paths

# %% codecell
data_dir = '../../../data/fig_1'
output_dir = '../../../outputs/fig_1'
pipeline_path = '../..'


# %% codecell
# params
resolution = 0.04
channel_names = ['rRNA','GFP_DNA']

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%load_ext autoreload
%autoreload 2

sys.path.append(pipeline_path + '/functions')
import fn_general_use as fgu
import image_plots as ip
import segmentation_func as sf
import fn_hiprfish_classifier as fhc
import fn_spectral_images as fsi

# %% md

# ==============================================================================
# ## Pick a sample
# ==============================================================================

# %% codecell
# get samples
ext = '.czi'
filenames = glob.glob(data_dir + '/*' + ext)
sample_names = [re.sub(ext,'',os.path.split(fn)[1]) for fn in filenames]
sample_names

# %% codecell
javabridge.start_vm(class_path=bioformats.JARS)

# %% codecell
# pick sample
i = 1
sn = sample_names[i]
fn = filenames[i]
raw = bioformats.load_image(fn)
raw_chans = [raw[:,:,i] for i in range(raw.shape[2])]

# %% codecell
# metadata
raw_aplc = aplc.CziFile(fn)

for elem in raw_aplc.meta.iter():
    if len(elem.attrib) > 0:
        print(elem.tag, elem.attrib)

# %% codecell
# Load cell morphology info
cell_props_fn = output_dir + '/segmentation/cell_seg_props/' + sn + '/' + sn + '_chan_0_cell_seg_props.csv'
cell_props = pd.read_csv(cell_props_fn)
cell_props.columns

# %% codecell
# Look at cell segmentation
cell_seg_fn = output_dir + '/segmentation/cell_seg/' + sn + '/' + sn + '_chan_0_cell_seg.npy'
cell_seg = np.load(cell_seg_fn)
# %% codecell
cell_seg_rgb = ip.seg2rgb(cell_seg)
ip.plot_image(cell_seg_rgb)

# %% codecell
c = [500, 500]
d = [500,500]
cell_seg_rgb_zoom = cell_seg_rgb[c[0]: c[0]+d[0], c[1]: c[1]+d[1], :]
ip.plot_image(cell_seg_rgb_zoom)

# %% codecell
import math

def magnitude(vector):
    return math.sqrt(sum(a * a for a in vector))


# %% codecell
a = [2,2]
b = [0,-1]

abs(np.dot(a,b) / magnitude(b))


# %% codecell
cell_props.columns

cell_props.loc[:5,'orientation']

# %% codecell
# Get spot props
spot_props_fn = output_dir + '/segmentation/spot_analysis/' + sn + '/' + sn + '_cellchan_0_spotchan_1_spot_seg_cid.csv'
spot_props = pd.read_csv(spot_props_fn)
spot_props.columns

# %% codecell
# Build cell dictionary to speed up lookups
cell_dict = {}
for i, c in cell_props.iterrows():
    l = c.label
    cell_dict[l] = {}
    cell_dict[l]['cent'] = np.array(eval(c.centroid))
    cell_dict[l]['ori'] = c.orientation
    cell_dict[l]['mal'] = c.major_axis_length
    cell_dict[l]['ecc'] = c.eccentricity

# %% codecell
# for each spot
pole_fracs = []
for i, sp in tqdm(spot_props.iterrows()):
    # get coords
    # get cell
    cid = sp.cell_id
    if cid > 0:
        cell = cell_dict[cid]
        # get orientation
        ori = cell['ori']
        # Get cell axis vector
        if abs(ori) == (math.pi / 2):
            vcell = [0,1]
        else:
            vcell = [1, math.tan(ori)]
        vcell = np.array(vcell)
        # Get spot axis vector
        sp_coords = np.array([sp['centroid-0'], sp['centroid-1']])
        vspot = sp_coords - cell['cent']
        # get dot product for spot
        dist = abs(np.dot(vspot, vcell) / magnitude(vcell))
        # Normalize by cell length
        cell_len = cell['mal'] / 2
        pole_fracs.append(dist / cell_len)

# %% codecell
n, bins, patches = plt.hist(pole_fracs, bins=200)


# %% codecell
hist = plt.hist(cell_props.eccentricity.values, bins=100)

# %% codecell
# only look at cells with eccentricity
ecc_thresh=0.9

pole_fracs = []
for i, sp in tqdm(spot_props.iterrows()):
    # get coords
    # get cell
    cid = sp.cell_id
    if cid > 0:
        cell = cell_dict[cid]
        if cell['ecc'] > ecc_thresh:
            # get orientation
            ori = cell['ori']
            # Get cell axis vector
            if abs(ori) == (math.pi / 2):
                vcell = [0,1]
            else:
                vcell = [1, math.tan(ori)]
            vcell = np.array(vcell)
            # Get spot axis vector
            sp_coords = np.array([sp['centroid-0'], sp['centroid-1']])
            vspot = sp_coords - cell['cent']
            # get dot product for spot
            dist = abs(np.dot(vspot, vcell) / magnitude(vcell))
            # Normalize by cell length
            cell_len = cell['mal'] / 2
            pole_fracs.append(dist / cell_len)

# %% codecell
n, bins, patches = plt.hist(pole_fracs, bins=100)




# %% codecell
