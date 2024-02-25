# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2021_11_09
# =============================================================================
"""
The notebook Has Been Built for...analysis of plaque positive and negative
controls: hao's plaque (HS) with plasmids, Ben's plaque (BMG) without plasmids.
Extract segmentation and spot counts to later analyze the spatial heterogenity
Airyscan images

For use with the hiprfish_imaging_py38 conda env
"""
# %% codecell
# =============================================================================
# Setup
# =============================================================================
image_analysis_code_dir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code'
nb_dir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/plaque/experiments/2021_08_26_plaqueamrintegrated/notebooks'
data_dir = '../data'
fig_dir = '../figures'
image_processing_dir = '../image_processing'
sample_names = [
        '2021_08_26_plaqueamrintegrated_probe_plasmid_sample_bmg_rep_2_fov_tile1',
        '2021_08_26_plaqueamrintegrated_probe_plasmid_sample_hs_rep_2_fov_tile1'
        ]
exts = ['_Airyscan Processing_shad_stitch.czi']*2
channel_names = ['smeS','MefE','mexZ','16s rRNA']

# %% codecell
# Imports
import sys
sys.path.append(image_analysis_code_dir)
import gc
import os
from czifile import CziFile
import image_plots as ip
from scipy.ndimage import gaussian_filter
import functions.segmentation_func as sf
from skimage import color
from tqdm import tqdm
import random
from skimage.feature import peak_local_max
from matplotlib import cm
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops_table
import anndata as ad
import pandas as pd
import javabridge
import bioformats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import squidpy as sq
from matplotlib import cm
from copy import copy
import scanpy as sc
from numba import njit
import json

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
# Load in tiled image
filenames = [
        '{}/{}{}'.format(data_dir, sn, ext)
        for sn, ext in zip(sample_names, exts)
    ]
raw_images = []
javabridge.start_vm(class_path=bioformats.JARS)
raw_images = [bioformats.load_image(f) for f in filenames]
raw_image_fns = [
        '{}/{}_raw.npy'.format(image_processing_dir, sn)
        for sn in sample_names
    ]
for r, f in zip(raw_images, raw_images_fns):
    np.save(f, r)
del raw_images

# # subset
# raw_images = [raw_images[0][1000:1500,2000:2500,:], raw_images[1][6000:6500,8000:8500,:]]


# %% codecell
# =============================================================================
# Visualize raw images,
# =============================================================================
# %% codecell
# Pick channels and set clims
channel_index=[1,2,3]
clims_raw=[(0, 0.1), (0, 0.1), (0, 0.3)]
cmaps_raw = ['inferno','inferno','gray']


# %% codecell
# show raw images
for f in raw_image_fns:
    im = np.load(f)
    im_list = [im[:,:,j] for j in channel_index]
    fig, ax, cbars = ip.subplot_square_images(
            im_list,
            (1,len(channel_index)),
            im_inches=5,
            clims=clims_raw,
            cmaps=cmaps_raw
        )
    ip.plt.show()
    ip.plt.close()
    del(im, im_list)

# %% codecell
# cell channel
raw_images = [np.load(f) for f in raw_image_fns]
ip.plt.figure(figsize=(20,20))
ip.plt.imshow(raw_images[1][:,:,3], cmap='inferno')
# ip.plt.imshow(raw_images[1][0,0,0,0,3,0,:,:,0], cmap='inferno')

# %% codecell
# cell channel
ip.plt.figure(figsize=(20,20))
ip.plt.imshow(raw_images[0][:,:,3], cmap='inferno')

# %% codecell
# =============================================================================
# segment,
# =============================================================================
seg_channel_index=3
sigmas=(3.5, 2.5)
small_objects=20
bg_thresholds=[0.08,0.07]

# %% codecell
# Pick bg thresholds
# bg_thresholds=[3000,4000]
viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([0/256, 0/256, 0/256, 1])
newcolors[:1, :] = pink
newcmp = ListedColormap(newcolors)

cmaps = [newcmp]*4
ims_cell = [np.load(f)[:,:,seg_channel_index] for f in raw_image_fns]
ims_thresh = [im*(im>t) for im, t in zip(ims_cell, bg_thresholds)]
im_list = ims_cell + ims_thresh
fig, ax, cbars = ip.subplot_square_images(
        im_list,
        (len(raw_images),len(raw_images)),
        im_inches=5,
        cmaps=cmaps
    )
ip.plt.show()
ip.plt.close()

# %% codecell
# check segmentaiton on zoom
from skimage.restoration import denoise_nl_means
cmaps = ['inferno','inferno','']
plot_shape=(1,3)
i = 1
t = bg_thresholds[i]
rcs = (2000,3000)
l = 500
sg = sigmas[i]
im = np.load(raw_image_fns[i])[rcs[0]:rcs[0]+l,rcs[1]:rcs[1]+l,:]
imp_cells = im[:,:,seg_channel_index]
# imp_log = np.log10(imp_cells)
imp_gauss = gaussian_filter(imp_cells, sigma=sg)
# imp_denoise = denoise_nl_means(imp_cells, patch_size=3, patch_distance=7, h=0.1)
# imp_seg = sf.segment(imp_log, small_objects=20, bg_threshold=t)
imp_seg = sf.segment(imp_gauss, small_objects=20, bg_threshold=t)
# imp_seg = sf.segment(imp_denoise, small_objects=20, bg_threshold=t)
imp_seg_color = color.label2rgb(imp_seg, bg_label = 0, bg_color = (0,0,0))
# im_list = [imp_cells, imp_log, imp_seg_color]
im_list = [imp_cells, imp_gauss, imp_seg_color]
# im_list = [imp_cells, imp_denoise, imp_seg_color]
fig, ax, cbars = ip.subplot_square_images(
        im_list,
        plot_shape,
        im_inches=5,
        cmaps=cmaps
    )
ip.plt.show()
ip.plt.close()

# %% codecell
# segment cells
seg_fns = ['{}/{}_seg.npy'.format(image_processing_dir, sn) for sn in sample_names]
for f, t, sg, s_fn in tqdm(zip(raw_image_fns, bg_thresholds, sigmas, seg_fns)):
    imp_cells = np.load(f)[:,:,seg_channel_index]
    imp_gauss = gaussian_filter(imp_cells, sigma=sg)
    imp_seg = sf.segment(imp_gauss, small_objects=20, bg_threshold=t)
    np.save(s_fn, imp_seg)
    print('Wrote: ', imp_seg)
del(imp_seg, imp_gauss, imp_cells)

# %% codecell
# show seg full size and save rgb version of seg
cmaps = ['inferno','']
plot_shape = (1,2)
seg_rgb_fns = [
        '{}/{}_seg_rgb.npy'.format(image_processing_dir, sn)
        for sn in sample_names
    ]
for r_fn, s_fn, sc_fn in zip(raw_image_fns, seg_fns, seg_rgb_fns):
    im, seg = [np.load(f) for f in [r_fn, s_fn]]
    imp_seg_color = color.label2rgb(seg, bg_label = 0, bg_color = (0,0,0))
    np.save(sc_fn, imp_seg_color)
    print('Wrote: ', sc_fn)
    im_list = [im[:,:,seg_channel_index], imp_seg_color]
    ip.subplot_square_images(im_list, plot_shape, cmaps=cmaps)
    ip.plt.show()
    ip.plt.close()
del(im,seg, imp_seg_color)

# %% codecell
# Show seg zoom
edge_size=1000
zoom_coords = []
rcs = [(1000,2000),(6000,8000)]
for r_fn, sc_fn, (r,c) in zip(raw_image_fns, seg_rgb_fns, rcs):
    im, seg = [np.load(f) for f in [r_fn, sc_fn]]
    im_zoom = im[r:r+edge_size, c:c+edge_size, seg_channel_index]
    seg_zoom = seg[r:r+edge_size, c:c+edge_size, :]
    im_list = [im_zoom, seg_zoom]
    print([i.shape for i in im_list])
    ip.subplot_square_images(im_list, plot_shape, cmaps=cmaps)
    ip.plt.show()
    ip.plt.close()

# %% codecell
# =============================================================================
# assign spots to cells
# =============================================================================
spot_channels_index=[1,2]
spot_channels_names=[channel_names[i] for i in spot_channels_index]

# %% codecell
# Cell info
properties = (
        'label','centroid','area','convex_area','perimeter','solidity',
        'major_axis_length','minor_axis_length','eccentricity','orientation',
        'max_intensity','mean_intensity'
    )
seg_info_fns = [
        '{}/{}_seg_info.csv'.format(image_processing_dir, sn)
        for sn in sample_names
    ]
for s_fn, r_fn, si_fn in zip(seg_fns, raw_image_fns, seg_info_fns):
    imp_cells = np.load(r_fn)[:,:,seg_channel_index]
    seg = np.load(s_fn)
    obs = pd.DataFrame(regionprops_table(
            seg,
            intensity_image=imp_cells,
            properties=properties
        ))
    obs = obs.set_index('label')
    obs.to_csv(si_fn)
    print('Wrote: ', si_fn)
del(obs,seg,imp_cells)

# %% codecell
# get local maxima coords, intensity, and cell assignment
columns=['spot_coord_0','spot_coord_1','intensity','cell_assign']
spot_int_cellassign_fns = [
        ['{}/{}_gene_{}_spot_int_cellassign.csv'.format(image_processing_dir, sn, g)
        for g in spot_channels_names]
        for sn in sample_names
    ]
for r_fn, s_fn, sic_fns in zip(raw_image_fns, seg_fns, spot_int_cellassign_fns):
    im, seg = [np.load(f) for f in [r_fn, s_fn]]
    for index, sic_fn in zip(spot_channels_index, sic_fns):
        im_ = im[:,:,index]
        m_i = peak_local_max(im_, min_distance=2, indices=True)
        int_i = np.array([im_[j,k] for j,k in m_i])[np.newaxis].T
        ca_i = np.array([seg[m[0],m[1]] for m in m_i])[np.newaxis].T
        mic = np.hstack([m_i, int_i, ca_i])
        mic_df = pd.DataFrame(mic, columns=columns)
        mic_df.to_csv(sic_fn)
        print('Wrote: ', sic_fn)
del(im, seg, im_, m_i, int_i, ca_i, mic, mic_df, m_int_ca)



# %% codecell
# =============================================================================
# Threshold spots, get counts, and get spot stats for each cell
# =============================================================================
count_thresholds = [(0.05,0.025)]*2
measures = ['_count','_max','_mean','_stdev']
X_columns=[gene + meas for gene in spot_channels_names for meas in measures]
morans_I_columns = [gene + '_count' for gene in spot_channels_names]

# %% codecell
# Pick count threshold
for sic_fns, c_t in zip(spot_int_cellassign_fns, count_thresholds):
    for sic_fn, ct in zip(sic_fns, c_t):
        ints = mic.intensity.values
        fig, a1 = plt.subplots(1,1, figsize=(10,10))
        threshs = np.linspace(np.min(ints), np.max(ints),1000)
        counts = [np.sum(ints > t) for t in threshs]
        len(counts)
        a1.plot(threshs, counts)
        x = [ct,ct]
        y = [np.min(counts), np.max(counts)]
        a1.plot(x,y)
        # a1.set_xlim(0,0.3)
        # a1.set_ylim(0,10000)
        plt.show()
        plt.close()

# %% codecell
# Calculate spot counts and max intensity, mean intensity, and stdev intensity for cellxvar
cell_info_spot_count_fns = [
        '{}/{}_seg_info_spot_count.csv'.format(image_processing_dir, sn)
        for sn in sample_names
    ]
zoom_coords = ((1500,2000, 1500,2000), (2000,2500,2000,2500))
# for sic_fns, si_fn, c_thr, cis_fn, zc in zip(spot_int_cellassign_fns, seg_info_fns, count_thresholds, cell_info_spot_count_fns, zoom_coords):
for sic_fns, si_fn, c_thr, cis_fn in zip(spot_int_cellassign_fns, seg_info_fns, count_thresholds, cell_info_spot_count_fns):
    # load cell info
    cell_info = pd.read_csv(si_fn)
    # bool_r = (cell_info['centroid-1'] > zc[0]) & (cell_info['centroid-1'] < zc[1])
    # bool_c = (cell_info['centroid-0'] > zc[2]) & (cell_info['centroid-0'] < zc[3])
    # cell_info = cell_info[bool_c & bool_r]
    for sic_fn, t, cn in zip(sic_fns, c_thr, spot_channels_names):
        # load spot df with cell assignment
        sic = pd.read_csv(sic_fn)
        # set up the acruing lists
        count, max, mean, stdev = [],[],[],[]
        # iterate through the cells
        for cid in tqdm(cell_info.label.values):
            temp_df = sic[sic.cell_assign == cid]
            # threshold the spots
            temp_df = temp_df[temp_df.intensity > t]
            # get spot stats for cell
            count.append(temp_df.shape[0])
            max.append(temp_df.intensity.max())
            mean.append(temp_df.intensity.mean())
            stdev.append(temp_df.intensity.std())
        cell_info[cn + '_count'] = count
        cell_info[cn + '_max'] =  np.nan_to_num(max, posinf=0, neginf=0)
        cell_info[cn + '_mean'] =  np.nan_to_num(mean, posinf=0, neginf=0)
        cell_info[cn + '_stdev'] =  np.nan_to_num(stdev, posinf=0, neginf=0)
    cell_info.to_csv(cis_fn)
    print('Wrote: ', cis_fn)

# %% codecell
# Recolor seg with stat values
seg_count_fns = [
        ['{}/{}_seg_gene_{}_count.npy'.format(image_processing_dir, sn, g)
        for g in spot_channels_names]
        for sn in sample_names
    ]
seg_max_fns = [
        ['{}/{}_seg_gene_{}_max.npy'.format(image_processing_dir, sn, g)
        for g in spot_channels_names]
        for sn in sample_names
    ]
# for cis_fn, s_fn, sc_fns, sm_fns, zc in zip(cell_info_spot_count_fns, seg_fns, seg_count_fns, seg_max_fns, zoom_coords):
for cis_fn, s_fn, sc_fns, sm_fns in zip(cell_info_spot_count_fns, seg_fns, seg_count_fns, seg_max_fns):
    print(cis_fn, s_fn)
    cis = pd.read_csv(cis_fn)
    seg = np.load(s_fn)
    # seg = seg[zc[0]:zc[1],zc[2]:zc[3]]
    # ids_ = np.unique(seg)
    # cis_sub = cis[cis.label.isin(ids_)]
    # ids = cis_sub.label.values
    ids = cis.label.values
    for gene, sc_fn, sm_fn in zip(spot_channels_names, sc_fns, sm_fns):
        # count recolor
        counts = cis[gene + '_count'].values
        # counts = cis_sub[gene + '_count'].values
        print(np.min(counts))
        print(np.max(counts))
        seg_count = ip.nb_recolor_seg(seg, ids=ids, vals=counts)
        print(np.nanmin(seg_count))
        print(np.nanmax(seg_count))
        sv = np.save(sc_fn, seg_count)
        print('Wrote: ', sc_fn)
        # max recolor
        # maxs = cis_sub[gene + '_max'].values
        maxs = cis[gene + '_max'].values
        print(np.min(maxs))
        print(np.max(maxs))
        seg_max = ip.nb_recolor_seg(seg, ids=ids, vals=maxs)
        print(np.nanmin(seg_max))
        print(np.nanmax(seg_max))
        sv = np.save(sm_fn, seg_max)
        print('Wrote: ', sm_fn)

# %% codecell
cis_fn = '../image_processing/2021_08_26_plaqueamrintegrated_probe_plasmid_sample_bmg_rep_2_fov_tile1_seg_info_spot_count.csv'
s_fn = '../image_processing/2021_08_26_plaqueamrintegrated_probe_plasmid_sample_bmg_rep_2_fov_tile1_seg.npy'
cis = pd.read_csv(cis_fn)
seg = np.load(s_fn)
zc = zoom_coords[0]
seg = seg[zc[0]:zc[1],zc[2]:zc[3]]
gene = 'mexZ'
ids_ = np.unique(seg)
cis_sub = cis[cis.label.isin(ids_)]
ids = cis_sub.label.values
counts = cis_sub[gene + '_count'].values
print(np.min(counts))
print(np.max(counts))
counts[counts==3].shape
seg_count = ip.nb_recolor_seg(seg, ids=ids, vals=counts)
print(np.nanmin(seg_count))
print(np.nanmax(seg_count))
cmap = ip.get_black_nan_cmap('viridis')
plt.imshow(seg_count, cmap=cmap)


# %% codecell


# %% codecell
# =============================================================================
# Save pngs and pdfs of segs
# =============================================================================

# %% codecell
# Save segmentation figure
seg_rgb_plot_fnt = fig_dir + '/{}_seg{}'
exts = ['.png','.pdf']
im_inches = 5
scalebar_resolution=0.035332945
for sn, s_fn in zip(sample_names, seg_rgb_fns):
    im = np.load(s_fn)
    fig, ax, cbar = ip.plot_image(im, scalebar_resolution=scalebar_resolution)
    for ext in exts:
        out_fn = seg_rgb_plot_fnt.format(sn,ext)
        plt.savefig(out_fn, dpi=int(np.max(im.shape)/im_inches))
        print('Wrote: ', out_fn)
    plt.show()
    plt.close()


# %% codecell
# Save counts seg
in_fns = seg_count_fns
seg_counts_plot_fnt = fig_dir + '/{}_seg_gene_{}_counts{}'
seg_counts_plot_cbar_fnt = fig_dir + '/{}_seg_gene_{}_counts_cbar{}'
out_fnt = seg_counts_plot_fnt
cb_fnt = seg_counts_plot_cbar_fnt
exts = ['.png','.pdf']
im_inches = 5
cmap = 'plasma'
clims = (0,4)
scalebar_resolution=0.035332945
for sn, s_fns in zip(sample_names, in_fns):
    for s_fn, gene in zip(s_fns, spot_channels_names):
        im = np.load(s_fn)
        cmap = ip.get_black_nan_cmap(cmap)
        fig, ax, cbar = ip.plot_image(
                im, im_inches=im_inches, cmap=cmap, discrete=True, clims=clims,
                scalebar_resolution=scalebar_resolution
            )
        for ext in exts:
            out_fn = out_fnt.format(sn,gene,ext)
            fig.savefig(out_fn, dpi=int(np.max(im.shape)/im_inches))
            print('Wrote: ', out_fn)
            cb_fn = cb_fnt.format(sn,gene,ext)
            cbar.savefig(cb_fn, bbox_inches='tight')
            print('Wrote: ', cb_fn)
        plt.show()
        plt.close()

# %% codecell
# Save maxs seg
in_fns = seg_max_fns
seg_max_plot_fnt = fig_dir + '/{}_seg_gene_{}_max{}'
seg_max_plot_cbar_fnt = fig_dir + '/{}_seg_gene_{}_max_cbar{}'
out_fnt = seg_max_plot_fnt
cb_fnt = seg_max_plot_cbar_fnt
exts = ['.png','.pdf']
im_inches = 5
cmap = 'plasma'
clims = (0,0.1)
scalebar_resolution=0.035332945
for sn, s_fns in zip(sample_names, in_fns):
    for s_fn, gene in zip(s_fns, spot_channels_names):
        im = np.load(s_fn)
        cmap = ip.get_black_nan_cmap(cmap)
        fig, ax, cbar = ip.plot_image(im, im_inches=im_inches, cmap=cmap, clims=clims, scalebar_resolution=scalebar_resolution)
        for ext in exts:
            out_fn = out_fnt.format(sn,gene,ext)
            fig.savefig(out_fn, dpi=int(np.max(im.shape)/im_inches))
            print('Wrote: ', out_fn)
            cb_fn = cb_fnt.format(sn,gene,ext)
            cbar.savefig(cb_fn)
            print('Wrote: ', cb_fn)
        plt.show()
        plt.close()





























# # %% codecell
# # Get cell unstructured info with coords and intensity for cell spot
# cell_gene_dict_fns = [
#         '{}/{}_cell_gene_dicts.json'.format(image_processing_dir, sn)
#         for sn in sample_names
#     ]
# for si_fn, sic_fns, cgd_fn in zip(
#         seg_info_fns, spot_int_cellassign_fns, cell_gene_dict_fns
#     ):
#     uns = {}
#     obs = pd.read_csv(si_fn)
#     sic_dfs = [pd.read_csv(f) for f in sic_fns]
#     for cid in tqdm(obs.index.values):
#         cid = str(cid)
#         uns[cid] = {}
#         for sic, cn in zip(sic_dfs, spot_channels_names):
#             uns[cid][cn] = sic[sic.cell_assign == cid]
#     file = open(cgd_fn, 'w')
#     json.dump(uns, file)
#     print('Wrote: ', cgd_fn)
# del(uns,obs,sic)

# # %% codecell
# # Calculate spot counts and max intensity, mean intensity, and stdev intensity for cellxvar
# image_cellxspotcount = []
# for uns, c_thresh, obs in zip(cell_gene_spot_info, count_thresholds, seg_info):
#     X = []
#     for cn, thresh in zip(spot_channels_names, c_thresh):
#         cid_count, cid_int_max, cid_int_mean, cid_int_stdev=[],[],[],[]
#         for cid in tqdm(obs.index.values):
#             cid_df = uns[cid][cn]
#             cid_df_filt = cid_df[cid_df.intensity > thresh]
#             cid_count.append(cid_df_filt.shape[0])
#             cid_int_max.append(cid_df_filt.intensity.max())
#             cid_int_mean.append(cid_df_filt.intensity.mean())
#             cid_int_stdev.append(cid_df_filt.intensity.std())
#         for c in [cid_count, cid_int_max, cid_int_mean, cid_int_stdev]:
#             X.append(c)
#     image_cellxspotcount.append(X)

# %% codecell
# Create anndata object and calculate stats
# image_adts = []
# for X, uns, obs, seg, seg_col, im in zip(image_cellxspotcount, cell_gene_spot_info, seg_info, segs, segs_color, raw_images):
#     # prep x dataframe
#     X_arr = np.array(X).T
#     X_df = pd.DataFrame(X_arr, columns=X_columns)
#     adt = ad.AnnData(X=X_df, obs=obs)
#     a = obs.loc[:,['centroid-1','centroid-0']].values
#     adt.obsm['spatial'] = a
#     adt.uns['cell_gene_spot_dfs'] = uns
#     adt.uns['segmentation'] = seg
#     adt.uns['segmentation_rgb'] = seg_col
#     adt.uns['raw_images'] = {channel_names[i]:im[:,:,i] for i in range(len(channel_names))}
#     # Spatial connectivity graph
#     W = sq.gr.spatial_neighbors(adt)
#     # Global moran's I
#     genes = []
#     sq.gr.spatial_autocorr(
#             adt,
#             connectivity_key='spatial_connectivities',
#             genes=morans_I_columns,
#             mode='moran',
#             n_perms=100
#         )
#     # Local cell stats
#     for col in morans_I_columns:
#         # Spatial Lag
#         W = adt.obsp['spatial_connectivities']
#         x_j = adt[:,col].X
#         lag = W*x_j
#         adt.obs[col + '_spatial_lag'] = lag
#         adt.obs[col + '_spatial_lag_quartile'] = pd.qcut(
#                 adt.obs[col + '_spatial_lag'],
#                 4,
#                 duplicates='drop',
#                 labels=False
#             )
#         # local moran's I
#         N = adt.obs_names.shape[0]
#         x_bar = np.mean(x_j)
#         x_i = np.array(x_j)
#         morans_i_loc = N*(x_i - x_bar)*(W*x_j)/(np.sum((x_i-x_bar)**2) + 1e-15)
#         adt.obs[col + '_morans_i_loc'] = morans_i_loc
#     image_adts.append(adt)
# image_adts[0]

# %% codecell
# =============================================================================
# Autocorrelation calculations
# =============================================================================

# %% codecell
# get spatial weights matrix


# %% codecell
# Calculate global moran's I

# %% codecell
# Calculate local moran's I


# %% codecell
# =============================================================================
# Plot stats
# =============================================================================
var_stat = '_count'
cell_cname = '16s rRNA'

# %% codecell
# Graph
for adt in image_adts:
    for gene in spot_channels_names:
        _, idx = adt.obsp["spatial_connectivities"][50,:].nonzero()
        idx = np.append(idx,50)
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        sc.pl.spatial(
                adt,
                neighbors_key="spatial_neighbors",
                color=gene + var_stat,
                edges=True,
                edges_width=1,
                img_key=1,
                spot_size=15,
                ax=ax,
                vmax=2
            )


# %% codecell
def recolor_seg_0(seg, vals_srs):
    seg_r = np.zeros(seg.shape)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            id = seg[i,j]
            seg_r[i,j] = vals_srs[id] if id else np.nan
            # seg_r[i,j] = vals_srs[str(id)] if id else np.nan
    return seg_r

@njit
def nb_recolor_seg_0(seg, ids, vals):
    seg_r = np.zeros(seg.shape)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            id = seg[i,j]
            seg_r[i,j] = vals[ids==id][0] if id else np.nan
    return seg_r



def recolor_seg_1(seg, ids, vals):
    seg_r = np.zeros(seg.shape)
    for id, val in zip(ids, vals):
        seg_r[seg==id] = val
    return seg_r


@njit
def nb_recolor_seg_1(seg, ids, vals):
    seg_r = np.zeros(seg.shape)
    for id, val in zip(ids,vals):
        seg_r += (seg==id)*val
    return seg_r

# %% codecell
s = 100
n=10
seg = np.zeros((s,s))
for i in range(n):
    seg[i,i] = i
ids = np.arange(n)
vals = np.linspace(0,1,n)
vals_srs = pd.Series(data=vals, index=ids)

# %% codecell
a=recolor_seg_0(seg,vals_srs)
%timeit recolor_seg_0(seg,vals_srs)
# %% codecell
a=nb_recolor_seg_0(seg,ids,vals)
%timeit nb_recolor_seg_0(seg,ids,vals)
# %% codecell
a=nb_recolor_seg_1(seg,ids,vals)
%timeit nb_recolor_seg_1(seg,ids,vals)
# %% codecell
a=recolor_seg_1(seg, ids, vals)
%timeit recolor_seg_1(seg, ids, vals)

# %% codecell
# replace cell seg with cell stats
for i, adt in enumerate(image_adts):
    print(i)
    for gene in spot_channels_names:
        print(gene)
        seg = adt.uns['segmentation']
        ids = adt.obs.index.values.astype(int)
        # Get recolored spot counts image
        vals = adt.to_df()[gene + var_stat].values
        im_var = nb_recolor_seg_0(seg, ids, vals)
        adt.uns['seg_' + gene + var_stat] = im_var
        # spatial lag image
        vals = adt.obs[gene + var_stat + '_spatial_lag'].values
        im_sl_quart = nb_recolor_seg_0(seg, ids, vals)
        adt.uns['seg_' + gene + var_stat + '_spatial_lag'] = im_sl_quart
        # local moran's I image
        vals = adt.obs[gene + var_stat + '_morans_i_loc'].values
        im_mi = nb_recolor_seg_0(seg, ids, vals)
        adt.uns['seg_' + gene + var_stat + '_morans_I'] = im_mi
    image_adts[i] = adt

# %% codecell
# plot
d_cmaps = ['plasma','viridis','cividis']
d_cmap_0 = copy(cm.get_cmap(d_cmaps[0]))
d_cmap_0.set_bad('k')
d_cmap_1 = copy(cm.get_cmap(d_cmaps[1]))
d_cmap_1.set_bad('k')
d_cmap_2 = copy(cm.get_cmap(d_cmaps[2]))
d_cmap_2.set_bad('k')
clims=[(0,0.1),(0,0.3),(),(),(),()]
cmaps=['inferno','gray','',d_cmap_0, d_cmap_1, d_cmap_2]
discrete_cmaps = [0,0,0,d_cmaps[0],0,0]
subplot_dims=(2,3)
im_inches=10
scalebar_resolution=0.035332945
for i, adt in enumerate(image_adts):
    for gene in spot_channels_names:
        spot = adt.uns['raw_images'][gene]
        cell = adt.uns['raw_images'][cell_cname]
        seg_col = adt.uns['segmentation_rgb']
        im_var = adt.uns['seg_' + gene + var_stat]
        im_sl_quart = adt.uns['seg_' + gene + var_stat + '_spatial_lag']
        im_mi = adt.uns['seg_' + gene + var_stat + '_morans_I']
        im_list = [spot, cell, seg_col, im_var, im_sl_quart, im_mi]
        ip.subplot_square_images(
                im_list,
                subplot_dims=subplot_dims,
                im_inches=im_inches,
                cmaps=cmaps,
                clims=clims,
                scalebar_resolution=scalebar_resolution,
                discrete=discrete_cmaps
            )
        plt.show()
        plt.close()
