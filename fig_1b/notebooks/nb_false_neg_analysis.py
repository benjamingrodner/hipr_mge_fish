# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_09
# =============================================================================
"""
The notebook Has Been Built for...check out the false negative cells from the
E. coli GFP experiments

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
data_dir = 'segmentation/raw_npy'
ext = '.npy'
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
# Manually look at the spot-cell assignment for spots that should be assigned
# =============================================================================
# Get sample names
sample_names = imfn.get_sample_names(data_dir=data_dir, ext=ext)
sample_names.sort()
for i, s in enumerate(sample_names): print(i,':', s)
# %% codecell
# Pick a fov
i = 59
sn = sample_names[i]

# %% codecell
# Get spot raw
raw_fn = data_dir + '/' + sn + ext
raw = np.load(raw_fn)
raw_spot = raw[:,:,1]
# Get seg
seg_fn = seg_dir + '/cell_seg/' + sn + '_cell_seg.npy'
seg = np.load(seg_fn)
# Get spot params
sp_df_fn = seg_dir + '/spot_analysis/' + sn + '_max_props_cid.csv'
sp_df = pd.read_csv(sp_df_fn)
# Filter by distance and intensity
dist = 5
intensity = 0.005
sp_df_filt = sp_df[(sp_df.cell_dist <= dist) & (sp_df.intensity >= intensity)]
sp_df_filt.columns
# %% codecell
# plot raw spot image with spots assigned to cells and cell seg
fig, ax, cbar = ip.plot_image(raw_spot, cmap='inferno', im_inches=20, plot_cbar=False,
                              clims=(0,0.03))
ax.scatter(sp_df_filt.col,sp_df_filt.row, s=0.5, color=(0,1,0))
ax = ip.plot_seg_outline(ax, seg, col=(0,0.8,0.8))
out_basename = seg_dir + '/spot_analysis/' + sn + '_spot_assign'
ip.save_png_pdf(out_basename, bbox_inches=False)
ip.plt.close()
# %% codecell
# plot false negative rate as a funciton of threshold
x = np.linspace(0,0.01,100)
sp_df_cell = sp_df[(sp_df.cell_dist <= dist)]
cell_count = np.unique(seg).shape[0]
nsrs = [sp_df_cell.loc[(sp_df_cell.intensity < i),'cell_id'] for i in x]
psrs = [sp_df_cell.loc[(sp_df_cell.intensity >= i),'cell_id'] for i in x]
fnr = [ns.unique().shape[0] / cell_count for ns in nsrs]
plt.plot(x, fnr)

# %% codecell
# =============================================================================
# Check out ROC curve
# =============================================================================
# Load negative data
i = 55
sn = sample_names[i]
# %% codecell
# Get spot raw
raw_fn = data_dir + '/' + sn + ext
raw = np.load(raw_fn)
raw_spot = raw[:,:,1]
# Get seg
seg_fn = seg_dir + '/cell_seg/' + sn + '_cell_seg.npy'
seg = np.load(seg_fn)
# Get spot params
sp_df_fn = seg_dir + '/spot_analysis/' + sn + '_max_props_cid.csv'
sp_df = pd.read_csv(sp_df_fn)
# %% codecell
# calculate values
neg_sp_df_cell = sp_df[(sp_df.cell_dist <= dist)]
neg_cell_count = np.unique(seg).shape[0]
neg_psrs = [neg_sp_df_cell.loc[(neg_sp_df_cell.intensity > i),'cell_id'] for i in x]
neg_nsrs = [neg_sp_df_cell.loc[(neg_sp_df_cell.intensity =< i),'cell_id'] for i in x]
fpr = [nps.unique().shape[0] / neg_cell_count for nps in neg_psrs]
tpr = [ps.unique().shape[0] / cell_count for ps in psrs]
tnr = [nps.unique().shape[0] / neg_cell_count for nps in neg_nsrs]
# %% codecell
# plot
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')

# %% codecell
# look at precision and false ommission
prec = [ps.unique().shape[0] / (ps.unique().shape[0] + nps.unique().shape[0])
        for ps, nps in zip(psrs, neg_psrs)]
FOR = [ns.unique().shape[0] / (ns.unique().shape[0] + nns.unique().shape[0] + 1e-5)
        for ns, nns in zip(nsrs, neg_nsrs)]
plt.plot(x,prec,label='ppv')
plt.plot(x,FOR,label='FOR')
plt.xlabel('threshold')
plt.legend()
# %% codecell
# plot all
plt.plot(x, tpr, label='tpr')
plt.plot(x, fpr, label='fpr')
plt.plot(x, tnr, label='tnr')
plt.plot(x, fnr,label='fnr')
plt.plot(x, prec,label='ppv')
plt.legend()
plt.xlabel('threshold')

# %% codecell
# =============================================================================
# Plot curves for all methods
# =============================================================================
# Get keys                          TODO: fill out and troubleshoot
keys = [imfn.get_filename_keys(sn, factors) for sn in sample_names]
# %% codecell
# make dict
sn_dict = imfn.get_nested_dict(keys, sample_names, [0,1])
# iterate through methods
for
    # iterate through pos/neg
        # Combine fovs
            # Get raw
            raw_fn = data_dir + '/' + sn + ext
            raw = np.load(raw_fn)
            raw_spot = raw[:,:,1]
            # Get seg
            seg_fn = seg_dir + '/cell_seg/' + sn + '_cell_seg.npy'
            seg = np.load(seg_fn)
            # Get spot params
            sp_df_fn = seg_dir + '/spot_analysis/' + sn + '_max_props_cid.csv'
            sp_df = pd.read_csv(sp_df_fn)
        # Filter by distance and intensity
        dist = 5
        intensity = 0.005
        sp_df_filt = sp_df[(sp_df.cell_dist <= dist) & (sp_df.intensity >= intensity)]
        sp_df_filt.columns
        # Get threshold curves and cell counts
        x = np.linspace(0,0.01,100)
        sp_df_cell = sp_df[(sp_df.cell_dist <= dist)]
        cell_count = np.unique(seg).shape[0]
        nsrs = [sp_df_cell.loc[(sp_df_cell.intensity < i),'cell_id'] for i in x]
        psrs = [sp_df_cell.loc[(sp_df_cell.intensity >= i),'cell_id'] for i in x]
    # calculate values
    FNR = [ns.unique().shape[0] / cell_count for ns in nsrs]
    FPR = [nps.unique().shape[0] / neg_cell_count for nps in neg_psrs]
    TPR = [ps.unique().shape[0] / cell_count for ps in psrs]
    TNR = [nps.unique().shape[0] / neg_cell_count for nps in neg_nsrs]
    PPV = [ps.unique().shape[0] / (ps.unique().shape[0] + nps.unique().shape[0])
            for ps, nps in zip(psrs, neg_psrs)]
    FOR = [ns.unique().shape[0] / (ns.unique().shape[0] + nns.unique().shape[0] + 1e-5)
            for ns, nns in zip(nsrs, neg_nsrs)]
    # plot
    plt.plot(x,PPV,label='PPV')
    plt.plot(x,FOR,label='FOR')
    plt.plot(x, TPR, label='TPR')
    plt.plot(x, FPR, label='FPR')
    plt.plot(x, TNR, label='TNR')
    plt.plot(x, FNR,label='FNR')
    plt.plot(x, PPV,label='PPV')
    plt.legend()
    plt.xlabel('Threshold')

# %% codecell
# Plot ROCs                             TODO: write and make pretty
# iterate through methods

# %% codecell
# =============================================================================
# Investigate seg size to see if false negatives are too small
# =============================================================================
# Get distribution of cell sizes
# get distribution of false negative cell sizes
# Show seg with different size filters to pick the best one
