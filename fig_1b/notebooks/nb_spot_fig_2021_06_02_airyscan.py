# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_09
# =============================================================================
"""
The notebook Has Been Built for...generating figures for spot segmentation of
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
# data_dir = '../../2021_06_02_airyscan/data'
seg_dir = '../2021_06_02_airyscan/image_processing'
fig_dir = 'figures'

# %% codecell
# Imports
import sys
import os
import gc
import pandas as pd
import numpy as np
import matplotlib as plt

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
# Load samples
# =============================================================================
# %% codecell
# All sample names
sample_names = imfn.get_sample_names(data_dir=seg_dir, ext='_cell_seg.npy')
sample_names.sort()
sample_names
# %% codecell
# Test sample name
sample_names = [sample_names[8],sample_names[11]]
sample_names
# %% codecell
# Get sample keys
factors = ['method','plasmid','fov']
keys = [imfn.get_filename_keys(sn, factors) for sn in sample_names]
keys
# %% codecell
spot_chan = 1
raws, spot_raws, cell_segs, max_props = [],[],[],[]
for sn in sample_names:
    # Load raw spot image
    raw = np.load(seg_dir + '/raw_npy/' + sn + '.npy')
    raws.append(raw)
    spot_raws.append(raw[:,:,spot_chan])
    # load cell seg
    cell_segs.append(np.load(seg_dir + '/cell_seg/' + sn + '_cell_seg.npy'))
    # Load max props
    max_props.append(pd.read_csv(seg_dir + '/spot_analysis/' + sn
                                 + '_max_props_cid.csv'))

# %% codecell
# =============================================================================
# Plot pixel itensity histogram
# =============================================================================
# Mask raw spot image using cell seg
spot_pixels_masked = [sr[(cs>0)] for sr, cs in zip(spot_raws, cell_segs)]
# %% codecell
# Generate a dictionairy for keys and pixels
sp_dict = imfn.get_nested_dict(keys, spot_pixels_masked, [0,1])
sp_dict
# %% codecell
# Set up ordered lists of factors
I = ['pos','neg']
J = ['pos', 'neg']
# Group pixel values and generate histograms
rnge = (0, 0.5)
bins = 100
hist_dict = {i:{} for i in I}
for i in I:
    for j in J:
        pix_list = sp_dict[i][j]
        pix = np.array([])
        for p in pix_list:
            pix = np.concatenate([pix,p[1]])
        y, bin_edges = np.histogram(pix, range=rnge, bins=bins)
        hist_dict[i][j] = y

# %% codecell
# plot intensities
c_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_styles=['-',':']
line_cols=[c_cycle[1], c_cycle[0]]
yticks = [2,4,6]
ylims = (-0.5,7)
xticks=[0,]
ax_col='k'
# legend_labels=['Positive','Negative']
# dims=[5,2]
# lw=2
# ft=20
legend_labels=[]
lw = 1
ft = 5
dims = (0.9843 ,0.5906)
intensity_values = 0.5*(bin_edges[1:]+bin_edges[:-1]).flatten()
group_filenaming = ['gel','nogel']
pixel_hist_fnt = fig_dir + '/pixel_int_hist_method_{}'
for i, gfn in zip(I, group_filenaming):
    hists = [hist_dict[i][j] for j in J]
    ip.plot_line_histogram(x_vals=intensity_values, y_vals=hists,
                             line_styles=line_styles, legend_labels=legend_labels,
                             xticks=xticks, yticks=yticks, line_cols=line_cols,
                             ax_col=ax_col, dims=dims, lw=lw, ft=ft)
    ip.save_png_pdf(pixel_hist_fnt.format(gfn), bbox_inches=False)
    ip.plt.show()
    ip.plt.close()


# %% codecell
# =============================================================================
# Select filter for spot intensity and filter spots
# =============================================================================
# Remove maxs not associated with cells
max_dist = 5
max_props_cell = [mp[(mp.cell_dist <= max_dist)] for mp in max_props]
# Generate a dictionairy for keys and maxs
mp_dict = imfn.get_nested_dict(keys, max_props_cell, [0,1])
# %% codecell
# Get threshold curves
t_lims = (0,0.04)
threshs = np.linspace(t_lims[0],t_lims[1],500)
curves = [[mp[mp.intensity > t].shape[0] for t in threshs] for mp in max_props_cell]
curves_dict = imfn.get_nested_dict(keys,curves,[0,1])
curves_dict.keys()
[curves_dict[k].keys() for k in curves_dict.keys()]
# %% codecell
# Plot threshold curves
for i in I:
    fig, ax = ip.general_plot(col='k', dims=(10,3))
    for j in J:
        curve_list = curves_dict[i][j]
        for k, c in curve_list:
            ax.plot(threshs, c, label=", ".join(k), lw=3)
            ax.legend(fontsize=10)
    ax.set_xlim(0, 0.01)
    ax.set_ylim(-100, 20000)
    ip.plt.show()
    ip.plt.close()
# %% codecell
# Pick threshold
thresholds = (0.025, 0.015)
# Plot on the curves
for i, threshold in zip(I, thresholds):
    fig, ax = ip.general_plot(col='k', dims=(10,3))
    for j in J:
        curve_list = curves_dict[i][j]
        for k, c in curve_list:
            ax.plot(threshs, c, label=", ".join(k), lw=3)
            ax.legend(fontsize=10)
        ax.plot([threshold]*2, [0,100000],'-k')
    ax.set_xlim(0, 0.04)
    ax.set_ylim(-1, 200)
    ip.plt.show()
    ip.plt.close()

# %% codecell
# =============================================================================
# Plot spot histogram
# =============================================================================
# Group max values and generate histograms
rnge = (0, 0.5)
bins = 100
hist_dict = {i:{} for i in I}
for i in I:
    for j in J:
        m_list = mp_dict[i][j]
        ms = np.array([])
        for m in m_list:
            ms = np.concatenate([ms,m[1].intensity.values])
        y, bin_edges = np.histogram(ms, range=rnge, bins=bins)
        hist_dict[i][j] = y

# %% codecell
# plot intensities
c_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_styles=['-',':']
legend_labels=[]
# legend_labels=['Positive','Negative']
line_cols=[c_cycle[1], c_cycle[0]]
yticks = [1,2,3]
ylims = (-0.3,4)
xticks=[0,0.25,0.5]
ax_col='k'
dims = (0.9843 ,0.5906)
# dims=[5,2]
lw=1
ft=5
intensity_values = 0.5*(bin_edges[1:]+bin_edges[:-1]).flatten()
for i, threshold in zip(I, thresholds):
    hists = [hist_dict[i][j] for j in J]
    fig, ax= ip.plot_line_histogram(x_vals=intensity_values, y_vals=hists,
                             line_styles=line_styles, legend_labels=legend_labels,
                             xticks=xticks, yticks=yticks, line_cols=line_cols,
                             ax_col=ax_col, dims=dims, lw=lw, ft=ft, ylims=ylims)
    ax.plot([threshold]*2, [0,ax.get_ylim()[1]],'-k')
    ip.plt.show()
    ip.plt.close()


# %% codecell
# =============================================================================
# Count spots per cell and plot violins
# =============================================================================
# Filter spots by intensity
mp_dict_int_filt = {i:{j:[mp[1][mp[1].intensity > thresh] for mp in mp_dict[i][j]]
                    for j in J}
                    for i, thresh in zip(I, thresholds)}
# Count spots per cell
cell_ids = [np.unique(cs) for cs in cell_segs]
cid_dict = imfn.get_nested_dict(keys, cell_ids, [0,1])
sc_dict = {}
for i in I:
    sc_dict[i] = {}
    for j in J:
        sc_dict[i][j] = []
        mps = mp_dict_int_filt[i][j]
        cids_l = cid_dict[i][j][1]
        spot_count = []
        for mp, cids in zip(mps, cids_l):
            for cid in cids:
                spot_count.append(mp[mp.cell_id == cid].shape[0])
        sc_dict[i][j] += spot_count
# %% codecell
# Plot violins
col = 'k'
lw=1
ft = 5
dims = (0.9843,0.5905512)
# dims = (3,1)
# dot_factor = 2
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_colors = [colors[0],colors[1]]
transparency = 0.8
fig_dir = '../figures'
plots = ['neg','pos']
groups = ['neg', 'pos']
xticklabels = ['Control', 'Positive']
xlabel = ''
ylabel = ''
jit = 0.15
y_jit = 0.1
ylims = (-0.3,10)
for pl in plots:
    fig, ax = ip.general_plot(col=col, dims=dims, lw=lw, ft=ft)
    xticks = ip.get_xticks(groups)
    ax.set_xticks(xticks)
    ax.set_xticklabels([],visible=False)
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_yticks([0,5,10])
    values = []
    for g,lc in zip(groups, line_colors):
        values.append(sc_dict[pl][g])
    ip.violin_dot_plot(ax, values, xticks, y_jit=y_jit, col=col, bw=0.5, lw=lw)
    ip.adjust_ylims(ax, lims=ylims, values=values, int_ticks=False)
