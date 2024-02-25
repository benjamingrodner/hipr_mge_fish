# %% md

# # Figure 5b: AMR Plasmid HiPR MGE FISH

# # version 00: visualize rois on raw and test registration on only one roi

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
from sklearn.neighbors import NearestNeighbors


gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5b'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %%

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

# %% md

# convert tiff files to npy and get a list of roi filenames

# %% codecell
# write roi tiffs to npy files
roi_dir = config['output_dir'] + '/roi_masks'
sn_hipr = '2022_03_19_plaquephagelytic_sample_bmg_probes_non_fov_tile1_round_2_mode_spec'
sn_mge = '20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_1_mode_airy_Airyscan_Processing_stitch'
sample_names = [sn_mge, sn_hipr]
roi_npy_fns = {}
roi_npy_labels = {}
# process for mge and hipr images
for sn in sample_names:
    roi_sn_dir = roi_dir + '/' + sn
    # Load tiffs
    roi_tiff_fns = glob.glob(roi_sn_dir + '/*.tif')
    roi_tiff_fns.sort()
    roi_npy_fns[sn] = []
    roi_npy_labels[sn] = []
    for fn in roi_tiff_fns:
        im = np.array(Image.open(fn))
        # save as numpy files
        out_fn = re.sub('.tif','.npy',fn)
        np.save(out_fn, im)
        roi_npy_fns[sn] += [out_fn]
        label = re.search(r'(?<=roi_)\d+', fn).group(0)
        roi_npy_labels[sn] += [label]

# %% md

# =============================================================================
# ## Show MGEFISH rois
# =============================================================================

# %% codecell
# Get MGE roi regionprops
mge_roi_props_list = []
# Get region properties
for i, fn in tqdm(enumerate(roi_npy_fns[sn_mge])):
    roi = 1*(np.load(fn) > 0)
    props_ = sf.measure_regionprops(roi, raw=np.zeros(roi.shape))
    mge_roi_props_list.append(props_)
    # plt.imshow(roi, interpolation='none')
    # plt.show()
    # plt.close()
    bbox = props_['bbox'].values[0]
    # print(bbox)
    roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    # plt.imshow(roi_box, interpolation='none')
    # plt.show()
    # plt.close()

mge_roi_props_df = pd.concat(mge_roi_props_list)
mge_roi_props_df_fn = config['output_dir'] + '/' + sn_mge + '_roi_props.csv'
mge_roi_props_df.to_csv(mge_roi_props_df_fn)

# %% codecell
# Get rois combined as outlines
r=10

mge_tiff_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_1_mode_airy_Airyscan_Processing_stitch_c1-2.tif'
mge_tiff = np.array(Image.open(mge_tiff_fn))
mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
mge_roi_overlay = np.zeros(mge_tiff.shape[:2])
for i, fn in tqdm(enumerate(roi_npy_fns[sn_mge])):
    # Load roi and props
    roi = 1*(np.load(fn) > 0)
    row = mge_roi_props_df.iloc[i,:]
    bbox = eval(row['bbox'])
    roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    roi_line = ip.find_boundaries(roi_box, mode='thick')
    roi_line_thick = spf.convolve_bool_circle(roi_line, r=r)[r:-r,r:-r]
    overl_box = mge_roi_overlay[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    overl_box = overl_box + roi_line_thick
    roi_line_thick.shape
    mge_roi_overlay[bbox[0]:bbox[2],bbox[1]:bbox[3]] = overl_box

mge_roi_overlay = 1*(mge_roi_overlay > 0)
plt.imshow(mge_roi_overlay, interpolation='none')
mge_roi_overlay_fn = config['output_dir'] + '/' + sn_mge + '_roi_overlay.npy'
np.save(output_fn, mge_roi_overlay_fn)

# %% codecell
# Show overlay on raw with markings
line_col = (1,1,0)
ft = 12
text_col = (0,1,1)
im_inches=10

fig, ax, cbar = ip.plot_image(mge_tiff, im_inches=im_inches)
mge_roi_overlay = np.load(mge_roi_overlay_fn)
mge_roi_overlay = mge_roi_overlay.astype(float)
mge_roi_overlay[mge_roi_overlay == 0] = np.nan
clims = (0,0.9)
cmap = ip.copy(plt.cm.get_cmap('gray'))
cmap.set_bad(alpha = 0)
cmap.set_over(line_col, 1.0)
ax.imshow(mge_roi_overlay, cmap=cmap, clim=clims, interpolation='none')
labels = [int(i) for i in roi_npy_labels[sn_mge]]
c = mge_roi_props_df['centroid'].values
for i, txt in enumerate(labels):
    c_ = eval(c[i])
    t = ax.text(c_[1], c_[0], s=txt, color=text_col, size=ft)
    t.set_bbox(dict(facecolor='k', alpha=0.5))
output_basename = config['output_dir'] + '/' + sn_mge + '_roi_overlay'
ip.save_png_pdf(output_basename)

# %% codecell
# Show overlay on raw with markings
line_col = (1,1,0)
ft = 12
text_col = (1,1,1)
im_inches=10
spot_col=(0,1,1)

# Plot image
fig, ax, cbar = ip.plot_image(mge_tiff, im_inches=im_inches)
# Plot filtered spots
spot_props_cid_fmt = config['output_dir'] + '/' + config['spot_props_cid_fmt']
sn_mge_clip = re.sub('_Airyscan_Processing_stitch','', sn_mge)
spot_props = pd.read_csv(
                spot_props_cid_fmt.format(sample_name=sn_mge_clip, cell_chan=0,spot_chan=1)
                )
sc = spot_props[spot_props.cell_id > 0].centroid.apply(lambda x: eval(x))
sc = np.array([[i for i in k] for k in sc])
ax.plot(sc[:,1], sc[:,0], '.', ms=4, color=spot_col)
# plot roi outlines
mge_roi_overlay = np.load(mge_roi_overlay_fn)
mge_roi_overlay = mge_roi_overlay.astype(float)
mge_roi_overlay[mge_roi_overlay == 0] = np.nan
clims = (0,0.9)
cmap = ip.copy(plt.cm.get_cmap('gray'))
cmap.set_bad(alpha = 0)
cmap.set_over(line_col, 1.0)
ax.imshow(mge_roi_overlay, cmap=cmap, clim=clims, interpolation='none')
# label ROIs
mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
labels = [int(i) for i in roi_npy_labels[sn_mge]]
c = mge_roi_props_df['centroid'].values
for i, txt in enumerate(labels):
    c_ = eval(c[i])
    t = ax.text(c_[1], c_[0], s=txt, color=text_col, size=ft)
    t.set_bbox(dict(facecolor='k', alpha=0.5))
# Save figure
output_basename = config['output_dir'] + '/' + sn_mge + '_roi_overlay_withspots '
ip.save_png_pdf(output_basename)

# %% codecell
# Plot cell image only with spot locations
clims=(0,0.1)

mge_raw_fn =  config['output_dir'] + '/raw_npy/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_1_mode_airy.npy'
mge_raw = np.load(mge_raw_fn)
mge_cell = mge_raw[:,:,0]
fig, ax, cbar = ip.plot_image(mge_cell, cmap='gray', clims=clims, im_inches=im_inches)
# Plot filtered spots
spot_props_cid_fmt = config['output_dir'] + '/' + config['spot_props_cid_fmt']
sn_mge_clip = re.sub('_Airyscan_Processing_stitch','', sn_mge)
spot_props = pd.read_csv(
                spot_props_cid_fmt.format(sample_name=sn_mge_clip, cell_chan=0,spot_chan=1)
                )
sc = spot_props[spot_props.cell_id > 0].centroid.apply(lambda x: eval(x))
sc = np.array([[i for i in k] for k in sc])
ax.plot(sc[:,1], sc[:,0], '.', ms=4, color=spot_col)
# plot roi outlines
mge_roi_overlay = np.load(mge_roi_overlay_fn)
mge_roi_overlay = mge_roi_overlay.astype(float)
mge_roi_overlay[mge_roi_overlay == 0] = np.nan
clims = (0,0.9)
cmap = ip.copy(plt.cm.get_cmap('gray'))
cmap.set_bad(alpha = 0)
cmap.set_over(line_col, 1.0)
ax.imshow(mge_roi_overlay, cmap=cmap, clim=clims, interpolation='none')
# label ROIs
mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
labels = [int(i) for i in roi_npy_labels[sn_mge]]
c = mge_roi_props_df['centroid'].values
for i, txt in enumerate(labels):
    c_ = eval(c[i])
    t = ax.text(c_[1], c_[0], s=txt, color=text_col, size=ft)
    t.set_bbox(dict(facecolor='k', alpha=0.5))
# Save figure
# output_basename = config['output_dir'] + '/' + sn_mge + '_roi_overlay_withspots_cellonly'
# ip.save_png_pdf(output_basename)


# %% md

# =============================================================================
# ## Show HiPRFISH rois
# =============================================================================


# %% codecell
# Get hiprfish roi regionprops
hipr_registered_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_2_mode_spec_registered.npy'
hipr_registered = np.load(hipr_registered_fn)
hipr_max = np.max(hipr_registered, axis=2)
hipr_roi_props_list = []
# Get region properties
for i, fn in tqdm(enumerate(roi_npy_fns[sn_hipr])):
    roi = np.load(fn)
    props_ = sf.measure_regionprops(roi, raw=hipr_max)
    hipr_roi_props_list.append(props_)

hipr_roi_props_df = pd.concat(hipr_roi_props_list)
hipr_roi_props_df_fn = config['output_dir'] + '/' + sn_hipr + '_roi_props.csv'
hipr_roi_props_df.to_csv(hipr_roi_props_df_fn)

# %% codecell
# Save overlays as outlines
r=5

hipr_roi_props_df = pd.read_csv(hipr_roi_props_df_fn)
hipr_roi_overlay = np.zeros(hipr_max.shape[:2])
for i, fn in tqdm(enumerate(roi_npy_fns[sn_hipr])):
    # Load roi and props
    roi = 1*(np.load(fn) > 0)
    row = hipr_roi_props_df.iloc[i,:]
    bbox = eval(row['bbox'])
    roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    roi_line = ip.find_boundaries(roi_box, mode='thick')
    roi_line_thick = spf.convolve_bool_circle(roi_line, r=r)[r:-r,r:-r]
    overl_box = hipr_roi_overlay[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    overl_box = overl_box + roi_line_thick
    roi_line_thick.shape
    hipr_roi_overlay[bbox[0]:bbox[2],bbox[1]:bbox[3]] = overl_box

hipr_roi_overlay = 1*(hipr_roi_overlay > 0)
plt.imshow(hipr_roi_overlay, interpolation='none')
hipr_roi_overlay_fn = config['output_dir'] + '/' + sn_hipr + '_roi_overlay.npy'
np.save(hipr_roi_overlay_fn, hipr_roi_overlay)

# %% codecell
# Show overlay on raw with markings
line_col = (1,1,0)
ft = 12
text_col = (0,1,1)
im_inches=10
clims=(0.005,0.25)

fig, ax, cbar = ip.plot_image(hipr_max, im_inches=im_inches, cmap='gray', clims=clims)
hipr_roi_overlay = np.load(hipr_roi_overlay_fn)
hipr_roi_overlay = hipr_roi_overlay.astype(float)
hipr_roi_overlay[hipr_roi_overlay == 0] = np.nan
clims = (0,0.9)
cmap = ip.copy(plt.cm.get_cmap('gray'))
cmap.set_bad(alpha = 0)
cmap.set_over(line_col, 1.0)
ax.imshow(hipr_roi_overlay, cmap=cmap, clim=clims, interpolation='none')
labels = [int(i) for i in roi_npy_labels[sn_hipr]]
hipr_roi_props_df = pd.read_csv(hipr_roi_props_df_fn)
c = hipr_roi_props_df['centroid'].values
for i, txt in enumerate(labels):
    c_ = eval(c[i])
    t = ax.text(c_[1], c_[0], s=txt, color=text_col, size=ft)
    t.set_bbox(dict(facecolor='k', alpha=0.5))
plt.sca(ax)
output_basename = config['output_dir'] + '/' + sn_hipr + '_roi_overlay'
ip.save_png_pdf(output_basename)


# %% codecell
# Show overlay on ident with markings
line_col = (1,1,1)
ft = 12
text_col = (1,1,1)
im_inches=10
clims=(0.005,0.25)

hipr_t20_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_2_mode_spec_ident_barcode.npy'
hipr_t20 = np.load(hipr_t20_fn)
fig, ax, cbar = ip.plot_image(hipr_t20, im_inches=im_inches, clims=clims)
hipr_roi_overlay = hipr_roi_overlay.astype(float)
hipr_roi_overlay[hipr_roi_overlay == 0] = np.nan
clims = (0,0.9)
cmap = ip.copy(plt.cm.get_cmap('gray'))
cmap.set_bad(alpha = 0)
cmap.set_over(line_col, 1.0)
ax.imshow(hipr_roi_overlay, cmap=cmap, clim=clims, interpolation='none')
labels = [int(i) for i in roi_npy_labels[sn_hipr]]
c = hipr_roi_props_df['centroid'].values
for i, txt in enumerate(labels):
    c_ = eval(c[i])
    t = ax.text(c_[1], c_[0], s=txt, color=text_col, size=ft)
    t.set_bbox(dict(facecolor='k', alpha=0.5))

output_basename = config['output_dir'] + '/' + sn_hipr + '_ident_roi_overlay'
ip.save_png_pdf(output_basename)



# %% md

# =============================================================================
# ## Get roi properties
# =============================================================================

# %% codecell

# %% codecell
# Get composition for each roi
# Load hiprfish identification
hipr_ident_bc_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_2_mode_spec_identification_barcode.npy'
hipr_ident_bc = np.load(hipr_ident_bc_fn)
hipr_seg_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_2_mode_spec_seg.npy'
hipr_seg = np.load(hipr_seg_fn)
hipr_roi_props_df = pd.read_csv(hipr_roi_props_df_fn)
hipr_roi_props_list = []
# Get region properties
for i, fn in tqdm(enumerate(roi_npy_fns[sn_hipr])):
    # Load roi and props
    roi = np.load(fn)
    row = hipr_roi_props_df.iloc[i,:]
    # Extract roi bbox region from bc image, roi, and seg
    bbox = eval(row['bbox'])
    roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    bc_box = hipr_ident_bc[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    seg_box = hipr_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    raw_max_bbox = hipr_max[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    bc_box_roi = bc_box * (roi_box > 0)
    seg_box_roi = seg_box * (roi_box > 0)
    np.unique(bc_box_roi)
    # Get regoin props for only roi in seg
    roi_props = sf.measure_regionprops(seg_box_roi, raw=raw_max_bbox)
    # Map labels to barcodes
    seg_bc_stack = np.dstack([seg_box_roi, bc_box_roi])
    seg_bc_map = np.unique(seg_bc_stack[seg_box_roi>0], axis=0)
    seg_bc_map_df = pd.DataFrame(seg_bc_map, columns=['label','barcode'])
    roi_props = roi_props.merge(seg_bc_map_df, how='left',on='label')
    # Save the roi props
    out_fn = re.sub('.npy','_ident_props.csv', fn)
    roi_props.to_csv(out_fn, index=False)

# %% codecell
# Construct composition dictionairy
taxon_color_lookup_fn = '/fs/cbsuvlaminck2/workdir/Data/HIPRFISH/Simulations/DSGN0672-0690/DSGN0673/taxon_color_lookup.csv'
taxon_color_lookup = pd.read_csv(taxon_color_lookup_fn)
taxon_color_lookup['code_b10'] = [int(str(c), base=2) for c in taxon_color_lookup.code.values]
taxon_color_lookup[:5]

# %% codecell
index_taxon_dict = dict(zip(
                            np.arange(taxon_color_lookup.shape[0]),
                            taxon_color_lookup['code_b10'].values
                            ))
roi_count_matrix = np.zeros((len(roi_npy_fns[sn_hipr]), len(index_taxon_dict)))
for i, fn in tqdm(enumerate(roi_npy_fns[sn_hipr])):
    roi_props_fn = re.sub('.npy','_ident_props.csv', fn)
    roi_props = pd.read_csv(roi_props_fn)
    roi_dict = roi_props['barcode'].astype(int).value_counts().to_dict()
    for index, taxon in index_taxon_dict.items():
        if taxon in roi_dict:
            roi_count_matrix[i,index] = roi_dict[taxon]

# %% codecell
# Normalize rows
row_sums = roi_count_matrix.sum(axis=1)
roi_count_matrix_rnorm = roi_count_matrix / row_sums[:, np.newaxis]
roi_count_matrix_rnorm.shape

# %% codecell
# UMAP embedding
reducer = umap.UMAP(random_state=42)
reducer.fit(roi_count_matrix_rnorm)
embedding = reducer.embedding_
fig, ax = plt.subplots()
ax.scatter(embedding[:,1], embedding[:,0])
n = np.arange(len(roi_npy_fns[sn_hipr])) + 1
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
for i, txt in enumerate(n):
    ax.annotate(txt, (embedding[i,1], embedding[i,0]))
ax.set_aspect('equal','box')
plt.sca(ax)
output_basename = config['output_dir'] + '/' + sn_hipr + '_UMAP'
ip.save_png_pdf(output_basename)

# %% codecell
# Get spot composition for each roi
# load segs
mge_cell_seg = ip.load_output_file(config, 'cell_seg_area_filt_fmt', sn_mge_clip,
                                    cell_chan=0, spot_chan=1)
mge_spot_seg = ip.load_output_file(config, 'spot_seg_cid_filt_fmt', sn_mge_clip,
                                    cell_chan=0, spot_chan=1)

mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
mge_roi_props_list = []
# Get region properties
for i, fn in tqdm(enumerate(roi_npy_fns[sn_mge])):
    # Load roi and props
    roi = np.load(fn)
    row = mge_roi_props_df.iloc[i,:]
    # Extract roi bbox region from bc image, roi, and seg
    bbox = eval(row['bbox'])
    roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    c_box = mge_cell_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    s_box = mge_spot_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    c_box_roi = c_box * (roi_box > 0)
    s_box_roi = s_box * (roi_box > 0)
    # Get counts
    c_count = np.unique(c_box_roi).shape[0]
    s_count = np.unique(s_box_roi).shape[0]
    mge_roi_props_list.append([c_count,s_count])
mge_roi_props_df = pd.DataFrame(mge_roi_props_list, columns=['cell_count','spot_count'])
mge_roi_props_fn = os.path.split(fn)[0] + '/roi_spot_cell_counts.csv'
mge_roi_props_df.to_csv(mge_roi_props_fn, index=False)

# %% codecell
# UMAP embedding colored by # spots normalized by number of cells
cmap_name = 'cool'
marker_size=20
dims=(7,5)
plot_ft=12
cbar_ft=10

cmap = plt.cm.get_cmap(cmap_name).copy()
for i in range(2):
    n = np.arange(len(roi_npy_fns[sn_hipr])) + 1
    mge_roi_props_df = pd.read_csv(mge_roi_props_fn)
    col_val = (mge_roi_props_df['spot_count'] / mge_roi_props_df['cell_count']).values
    if i == 0:
        col_val = np.pad(col_val, (0,embedding.shape[0]-col_val.shape[0]), 'constant',constant_values=(0,np.nan))
        p = plt.scatter(embedding[:,1], embedding[:,0], c=col_val, cmap=cmap)
        plt.gca().set_visible(False)
        cbar = plt.colorbar(p)
        cbar.ax.tick_params(labelsize=cbar_ft)
        output_basename = config['output_dir'] + '/' + sn_hipr + '_UMAP_spotcount_cbar'
    else:
        fig, ax = plt.subplots(figsize=dims)
        max = np.max(col_val)
        for i, txt in enumerate(n):
            try:
                cv = col_val[i] / max
                spot_color = cmap(cv)
            except:
                spot_color = (0.5,0.5,0.5,0.8)
            ax.plot(embedding[i,1], embedding[i,0],'.', ms=marker_size, color=spot_color)
            ax.annotate(txt, (embedding[i,1], embedding[i,0]))
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlabel('UMAP 1')
        # ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal', 'box')
        plt.sca(ax)
        output_basename = config['output_dir'] + '/' + sn_hipr + '_UMAP_spotcount'
    ip.save_png_pdf(output_basename)
    plt.show()
    plt.close()


# %% md

# =============================================================================
# ## Overlay ROIs
# =============================================================================

# %% md

# Extract ROIs that overlap well, resize and save as new images

# %% codecell
i = 11  # Test index roi
# for i in range(mge_roi_props_df.shape[0]):
mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
row = mge_roi_props_df.iloc[i,:]
bbox = eval(row['bbox'])
mge_roi_fn = roi_npy_fns[sn_mge][i]
mge_roi = np.load(mge_roi_fn)
mge_cell_bbox = mge_cell[bbox[0]:bbox[2],bbox[1]:bbox[3]]
mge_roi_bbox = mge_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
mge_cell_bbox_roi = mge_cell_bbox * mge_roi_bbox
output_fn = re.sub('.npy','_withraw.tiff', mge_roi_fn)
im = Image.fromarray(mge_cell_bbox_roi)
im.save(output_fn)
print('Wrote:', output_fn)
# plt.imshow(mge_cell_bbox_roi)
# plt.show()
# plt.close()


hipr_roi_props_df = pd.read_csv(hipr_roi_props_df_fn)
row = hipr_roi_props_df.iloc[i,:]
bbox = eval(row['bbox'])
hipr_roi_fn = roi_npy_fns[sn_hipr][i]
hipr_roi = np.load(hipr_roi_fn)
hipr_cell_bbox = hipr_max[bbox[0]:bbox[2],bbox[1]:bbox[3]]
hipr_roi_bbox = hipr_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
hipr_cell_bbox_roi = hipr_cell_bbox * hipr_roi_bbox
res_mge = 0.02995
res_hipr = 0.06748
factor_resize = res_hipr / res_mge
hipr_cell_bbox_roi_resize = resize(
                                hipr_cell_bbox_roi, None, fx = factor_resize,
                                fy = factor_resize, interpolation = INTER_NEAREST
                                )
output_fn = re.sub('.npy','_withraw_resize.tiff', hipr_roi_fn)
im = Image.fromarray(hipr_cell_bbox_roi_resize)
im.save(output_fn)
# print('Wrote:', output_fn)
# plt.imshow(hipr_cell_bbox_roi_resize)



# %% md

# Load ROIs in FIJI and manually select sister points for translation

# Now load points in python

# %% codecell
mge_pts_size=5
hipr_pts_size=5

# Show MGE points
mge_roi_pts_fn = re.sub('.npy','_points.csv',mge_roi_fn)
mge_roi_pts = pd.read_csv(mge_roi_pts_fn)[['X','Y']].values
fig, ax, cbar = ip.plot_image(mge_cell_bbox_roi)
ax.plot(mge_roi_pts[:,0], mge_roi_pts[:,1], '.r', ms=mge_pts_size)
plt.sca(ax)
output_basename = re.sub('.npy','_register_points', mge_roi_fn)
ip.save_png_pdf(output_basename)
plt.show()
plt.close()
# Show HIpr points
hipr_roi_pts_fn = re.sub('.npy','_points.csv',hipr_roi_fn)
hipr_roi_pts = pd.read_csv(hipr_roi_pts_fn)[['X','Y']].values
fig, ax, cbar = ip.plot_image(hipr_cell_bbox_roi_resize)
ax.plot(hipr_roi_pts[:,0], hipr_roi_pts[:,1], '.r', ms=mge_pts_size)
plt.sca(ax)
output_basename = re.sub('.npy','_register_points', hipr_roi_fn)
ip.save_png_pdf(output_basename)
plt.show()
plt.close()


# %% md

# Translate images in x-y and rotation

# %% codecell
print(hipr_cell_bbox_roi_resize.shape)
print(mge_cell_bbox_roi.shape)
# %% codecell
size=(1372,1636)  # Pick size of image

# %% md

# Now warp the images

# %% codecell
mge_pts_weight = 0.8  # Fraction, how close to the mge pts is the dest of the transform

from fn_face_morpher import warp_image, weighted_average_points
result_points = weighted_average_points(mge_roi_pts, hipr_roi_pts, mge_pts_weight)
roi_mge_wrp = warp_image(mge_cell_bbox_roi, mge_roi_pts, result_points, size)
output_fn = re.sub('.npy','_warp.npy', mge_roi_fn)
np.save(output_fn, roi_mge_wrp)
print('Wrote:', output_fn)
roi_hipr_wrp = warp_image(hipr_cell_bbox_roi_resize, hipr_roi_pts, result_points, size)
output_fn = re.sub('.npy','_warp.npy', hipr_roi_fn)
np.save(output_fn, roi_hipr_wrp)
print('Wrote:', output_fn)

# %% codecell
# Show
im_list = [mge_cell_bbox_roi, roi_mge_wrp, hipr_cell_bbox_roi_resize, roi_hipr_wrp]
fig, ax, cbar = ip.subplot_square_images(im_list, (2,2))
plt.sca(ax)
output_basename = config['output_dir'] + '/' + sn_mge + '_warp'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()

# %% md

# Shift unaltered images to overlap

# %% codecell
# show pre-shift
fig, ax = ip.general_plot()
ax.plot(mge_roi_pts[:,0], mge_roi_pts[:,1], '.r', ms=mge_pts_size)
ax.plot(hipr_roi_pts[:,0], hipr_roi_pts[:,1], '.b', ms=mge_pts_size)
plt.show()


# %% codecell
# minimize distances
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
def sumdist(shift, pts1, pts2):
    pts2 = np.copy(pts2)
    for i, s in enumerate(shift):
        pts2[:,i] = pts2[:,i] + s
    D = np.sqrt(np.sum((pts1-pts2)**2, axis=1))
    return(np.sum(D), pts2)

pts1 = np.copy(mge_roi_pts)
pts2 = np.copy(hipr_roi_pts)

n = 100
min_diff = 2
diff = 1e10
val_init = sumdist((0,0), mge_roi_pts, hipr_roi_pts)[0]
val_old = val_init
step_reduce = 0.75
i = 0
step = np.min(np.abs(pts1-pts2))
shift_list = []
while (i < n) and (diff > min_diff):
    shifts = [[0,step],[0,-step],[step,0],[-step,0]]  # Rook movement options
    sd_pts = [sumdist(s, pts1, pts2) for s in shifts]  # evaluate moves
    sd = [s[0] for s in sd_pts]  # Extract values
    pts_all = [s[1] for s in sd_pts]  # Extract new points
    sd_ind = np.argmin(sd)  # Evaluate options
    val_new = sd[sd_ind]
    shift_choice = shifts[sd_ind]
    if val_new < val_old:  # Compare and replace values
        pts2 = pts_all[sd_ind]
        shift_list.append(shift_choice)
        diff = val_old - val_new
        val_old = val_new
    else:  # Reduce step size
        step = round(step*step_reduce)
    i += 1

shift_final = np.sum(np.array(shift_list), axis=0).astype(int)
print(i)
print(shift_list)
print(shift_final)

# %% codecell
# Check points overlap
fig, ax = ip.general_plot()
ax.plot(pts1[:,0], pts1[:,1], '.r', ms=mge_pts_size)
ax.plot(pts2[:,0], pts2[:,1], '.b', ms=mge_pts_size)
plt.show()
# %% codecell
# Shift whole image
row_man_shift = 100

shift_final[0] = shift_final[0] + row_man_shift
sf_max = np.max(np.abs(shift_final))
shape = np.array(mge_cell_bbox_roi.shape)
shape_new = shape + 2*sf_max
mcbr_shift = np.zeros(shape_new)
mcbr_shift[sf_max:-sf_max, sf_max:-sf_max] = mge_cell_bbox_roi
plt.imshow(mcbr_shift)
plt.show()

ul = shift_final + sf_max
lr = np.array(hipr_cell_bbox_roi_resize.shape) + ul
hcbrr_shift = np.zeros(shape_new)
hcbrr_shift[ul[0]:lr[0], ul[1]:lr[1]] = hipr_cell_bbox_roi_resize
plt.imshow(hcbrr_shift)
plt.show()


# %% md

# Compare overlap manual to face morph

# %% codecell
# Shift images
mcbr_norm = ip.zero_one_thresholding(mcbr_shift, clims=(0,0.7))
hcbrr_norm = ip.zero_one_thresholding(hcbrr_shift, clims=(0,1))
overlay_shift = np.dstack([mcbr_norm, hcbrr_norm, np.zeros(hcbrr_norm.shape)])
# morph images
roi_mge_wrp_norm = ip.zero_one_thresholding(roi_mge_wrp, clims=(0,0.7))
roi_hipr_wrp_norm = ip.zero_one_thresholding(roi_hipr_wrp, clims=(0,1))
overlay_warp = np.dstack([roi_mge_wrp_norm, roi_hipr_wrp_norm, np.zeros(roi_hipr_wrp_norm.shape)])

image_list = [
            mcbr_norm, hcbrr_norm, overlay_shift,
            roi_mge_wrp_norm, roi_hipr_wrp_norm, overlay_warp
            ]
cmaps = ['inferno','inferno','']*2
ip.subplot_square_images(image_list,(2,3),cmaps=cmaps)
output_basename = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_overlay_comparison'
ip.save_png_pdf(output_basename)
# fig, (a1,a2, a3) = plt.subplots(1,3, figsize=(30,20))
# a1.imshow(mcbr_norm, cmap='inferno')
# a2.imshow(hcbrr_norm, cmap='inferno')
# a3.imshow(overlay)

# %% md

# Get overlap on MGE-FISH spot seg and hiprfish ident

# %% codecell
# Get barcode ROI
hipr_bc_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_2_mode_spec_identification_barcode.npy'
hipr_bc = np.load(hipr_bc_fn)
np.unique(hipr_bc)
hipr_roi_props_df = pd.read_csv(hipr_roi_props_df_fn)
row = hipr_roi_props_df.iloc[i,:]
bbox = eval(row['bbox'])
hipr_roi_fn = roi_npy_fns[sn_hipr][i]
hipr_roi = np.load(hipr_roi_fn)
hipr_bc_bbox = hipr_bc[bbox[0]:bbox[2],bbox[1]:bbox[3]]
hipr_roi_bbox = hipr_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
hipr_bc_bbox_roi = hipr_bc_bbox * (hipr_roi_bbox > 0)
np.unique(hipr_bc_bbox_roi)

res_mge = 0.02995
res_hipr = 0.06748
factor_resize = res_hipr / res_mge
hipr_bc_bbox_roi_resize = resize(
                                hipr_bc_bbox_roi, None, fx = factor_resize,
                                fy = factor_resize, interpolation = INTER_NEAREST
                                )
output_fn = re.sub('.npy','_withbarcode.npy', mge_roi_fn)
np.save(output_fn, hipr_bc_bbox_roi_resize)
print('Wrote:', output_fn)
plt.imshow(hipr_bc_bbox_roi_resize)
plt.show()

# %% codecell
# Warp barcode ROI
hbbrr_wrp = warp_image(hipr_bc_bbox_roi_resize, hipr_roi_pts, result_points, size,
                        dtype=np.int32, interpolation='nearest')
output_fn = re.sub('.npy','_withbarcode_warped.npy', mge_roi_fn)
np.save(output_fn, hbbrr_wrp)
print('Wrote:', output_fn)
plt.imshow(hbbrr_wrp)

# %% codecell
# Recolor with t20
# Get counts
hipr_cell_info_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_2_mode_spec_cell_information_consensus.csv'
hipr_cell_info = pd.read_csv(hipr_cell_info_fn)
taxon_color_lookup_fn = '/fs/cbsuvlaminck2/workdir/Data/HIPRFISH/Simulations/DSGN0672-0690/DSGN0673/taxon_color_lookup.csv'
taxon_color_lookup = pd.read_csv(taxon_color_lookup_fn)
taxon_color_lookup[taxon_color_lookup.code==100011]
bc_counts = []
for bc in taxon_color_lookup.code.values:
    bc_counts.append(hipr_cell_info.loc[hipr_cell_info.cell_barcode == bc].shape[0])
taxon_color_lookup['counts'] = bc_counts
# sort by ascending abundance
taxon_color_lookup = taxon_color_lookup.sort_values('counts',ascending=False)
colors_tab20 = plt.get_cmap('tab20').colors
c_order_0 = np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()
# colors
n_bcs = barcodes.shape[0]
grey_index = 15
drop_index = 4
c_order_0.remove(grey_index)
c_order_0.remove(drop_index)
c_order_0 += [grey_index] * (n_bcs - 18)
cols_t20_ordered_0 = [colors_tab20[i] for i in c_order_0]
taxon_color_lookup['tab20'] = cols_t20_ordered_0 + [(0,0,0)]*(taxon_color_lookup.shape[0] - len(cols_t20_ordered_0))
# colormap
num_genera=18
names = taxon_color_lookup['sci_name'].values[:num_genera].tolist()
names += ['other']
counts = taxon_color_lookup['counts'].values[:num_genera].tolist()
counts += [taxon_color_lookup['counts'].values[num_genera:].sum()]
label_color='k'
ip.taxon_legend(names, cols_t20_ordered_0[:20], label_color=label_color, taxon_counts=counts)
output_fn = config['output_dir'] + '/taxon_legend'
ip.save_png_pdf(output_fn)
plt.show()
plt.close()

# %% codecell
# Plot legend
num_genera=19
names = taxon_color_lookup['sci_name'].values[:num_genera].tolist()
names += ['other']
counts = taxon_color_lookup['counts'].values[:num_genera].tolist()
counts += [taxon_color_lookup['counts'].values[num_genera:].sum()]
label_color='k'
ip.taxon_legend(names, cols_t20_ordered_0[:20], label_color=label_color, taxon_counts=counts)
output_fn = config['output_dir'] + '/' + sn_hipr + '_ident_legend'
ip.save_png_pdf(output_fn)

# %% codecell
# Recolor roi
taxon_color_lookup['code_b10'] = [int(str(c), base=2) for c in taxon_color_lookup.code.values]
barcodes = taxon_color_lookup.loc[taxon_color_lookup.counts > 0, 'code_b10']
hbbrr_wrp_tab20 = ip.color_bc_image(hbbrr_wrp, barcodes, cols_t20_ordered_0)
output_fn = re.sub('.npy','_withbarcode_warped_t20.npy', mge_roi_fn)
np.save(output_fn, hbbrr_wrp_tab20)
print('Wrote:', output_fn)
plt.imshow(hbbrr_wrp_tab20)

# %% codecell
# Get spot seg ROI
mge_spot_seg = ip.load_output_file(config, 'spot_seg_max_split_fmt', sn_mge_clip,
                                    cell_chan=0, spot_chan=1)
mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
row = mge_roi_props_df.iloc[i,:]
bbox = eval(row['bbox'])
mge_roi_fn = roi_npy_fns[sn_mge][i]
mge_roi = np.load(mge_roi_fn)
mge_ss_bbox = mge_spot_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
mge_roi_bbox = mge_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
mge_ss_bbox_roi = mge_ss_bbox * mge_roi_bbox

output_fn = re.sub('.npy','_withspotseg.npy', mge_roi_fn)
np.save(output_fn, mge_ss_bbox_roi)
print('Wrote:', output_fn)
plt.imshow(mge_ss_bbox_roi)
plt.show()

# %% codecell
# Warp spot seg ROI
msbr_wrp = warp_image(mge_ss_bbox_roi, mge_roi_pts, result_points, size,
                        dtype=np.int32, interpolation='nearest')
output_fn = re.sub('.npy','_withspotseg_warped.npy', mge_roi_fn)
np.save(output_fn, msbr_wrp)
print('Wrote:', output_fn)
plt.imshow(msbr_wrp)

# %% codecell
# Get raw spot roi
mge_spot = mge_raw[:,:,1]
mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
row = mge_roi_props_df.iloc[i,:]
bbox = eval(row['bbox'])
mge_roi_fn = roi_npy_fns[sn_mge][i]
mge_roi = np.load(mge_roi_fn)
mge_spot_bbox = mge_spot[bbox[0]:bbox[2],bbox[1]:bbox[3]]
mge_roi_bbox = mge_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
mge_spot_bbox_roi = mge_spot_bbox * (mge_roi_bbox > 0)
output_fn = re.sub('.npy','_withrawspot.npy', mge_roi_fn)
np.save(output_fn, mge_spot_bbox_roi)
print('Wrote:', output_fn)
plt.imshow(mge_spot_bbox_roi)

# %% codecell
# Warp raw spot
mge_spot_bbox_roi_wrp = warp_image(mge_spot_bbox_roi, mge_roi_pts, result_points, size,
                        dtype=np.float64)
output_fn = re.sub('.npy','_withrawspot_warped.npy', mge_roi_fn)
np.save(output_fn, mge_spot_bbox_roi_wrp)
print('Wrote:', output_fn)
plt.imshow(mge_spot_bbox_roi_wrp)

# %% codecell
# Get roi props on warp image
msbr_wrp_props = sf.measure_regionprops(msbr_wrp, raw=mge_spot_bbox_roi_wrp)

# %% codecell
# overlay spots on hiprfish ident
im_inches=10

marker='.'
spot_col=np.array([0,1.,0,0.5])
marker_size=2000
linewidths=5

spot_int = msbr_wrp_props.max_intensity.values
spot_int /= np.max(spot_int)

marker_size = marker_size * spot_int
# spot_col = np.array([spot_col for i in range(msbr_wrp_props.shape[0])])
# spot_col[:,3] = alpha_vals.tolist()
fig, ax, cbar = ip.plot_image(hbbrr_wrp_tab20, im_inches=im_inches)
sc = msbr_wrp_props.centroid
sc = np.array([[i for i in k] for k in sc])
ax.scatter(sc[:,1], sc[:,0], marker=marker, s=marker_size, c=spot_col, linewidths=linewidths, edgecolors='none')
output_basename = re.sub('.npy','_warped_ident_spotoverlay', hipr_roi_fn)
ip.save_png_pdf(output_basename)
print('Wrote:', output_basename)

# %% codecell
# hiprfish ident
im_inches=10

marker='.'
spot_col=np.array([0,1.,0,0.5])
marker_size=2000
linewidths=5

spot_int = msbr_wrp_props.max_intensity.values
spot_int /= np.max(spot_int)

marker_size = marker_size * spot_int
# spot_col = np.array([spot_col for i in range(msbr_wrp_props.shape[0])])
# spot_col[:,3] = alpha_vals.tolist()
fig, ax, cbar = ip.plot_image(hbbrr_wrp_tab20, im_inches=im_inches)
sc = msbr_wrp_props.centroid
sc = np.array([[i for i in k] for k in sc])
# ax.scatter(sc[:,1], sc[:,0], marker=marker, s=marker_size, c=spot_col, linewidths=linewidths, edgecolors='none')
output_basename = re.sub('.npy','_warped_ident', hipr_roi_fn)
ip.save_png_pdf(output_basename)
print('Wrote:', output_basename)


# %% md

# =============================================================================
# ## Warp multiple rois
# =============================================================================

# Get rois with cell raw data and save to tiff file

# %% codecell
for i in range(mge_roi_props_df.shape[0]):
    mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
    row = mge_roi_props_df.iloc[i,:]
    bbox = eval(row['bbox'])
    mge_roi_fn = roi_npy_fns[sn_mge][i]
    mge_roi = np.load(mge_roi_fn)
    mge_cell_bbox = mge_cell[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    mge_roi_bbox = mge_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    mge_cell_bbox_roi = mge_cell_bbox * mge_roi_bbox
    output_fn = re.sub('.npy','_withraw.npy', mge_roi_fn)
    np.save(output_fn, mge_cell_bbox_roi)
    output_fn = re.sub('.npy','_withraw.tiff', mge_roi_fn)
    im = Image.fromarray(mge_cell_bbox_roi)
    im.save(output_fn)
    print('Wrote:', output_fn)
    # plt.imshow(mge_cell_bbox_roi)
    # plt.show()
    # plt.close()


    hipr_roi_props_df = pd.read_csv(hipr_roi_props_df_fn)
    row = hipr_roi_props_df.iloc[i,:]
    bbox = eval(row['bbox'])
    hipr_roi_fn = roi_npy_fns[sn_hipr][i]
    hipr_roi = np.load(hipr_roi_fn)
    hipr_cell_bbox = hipr_max[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    hipr_roi_bbox = hipr_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    hipr_cell_bbox_roi = hipr_cell_bbox * hipr_roi_bbox
    res_mge = 0.02995
    res_hipr = 0.06748
    factor_resize = res_hipr / res_mge
    hipr_cell_bbox_roi_resize = resize(
                                    hipr_cell_bbox_roi, None, fx = factor_resize,
                                    fy = factor_resize, interpolation = INTER_NEAREST
                                    )
    output_fn = re.sub('.npy','_withraw_resize.npy', hipr_roi_fn)
    np.save(output_fn, hipr_cell_bbox_roi_resize)
    output_fn = re.sub('.npy','_withraw_resize.tiff', hipr_roi_fn)
    im = Image.fromarray(hipr_cell_bbox_roi_resize)
    im.save(output_fn)

# %% md

# Now go pick sister points in FIJI ...

# ....

# Show picked points on raw images

# %% codecell
for mge_roi_fn, hipr_roi_fn in tqdm(zip(roi_npy_fns[sn_mge], roi_npy_fns[sn_hipr])):
    mge_roi_pts_fn = re.sub('.npy','_withraw_pts.csv',mge_roi_fn)
    hipr_roi_pts_fn = re.sub('.npy','_withraw_resize_pts.csv',hipr_roi_fn)
    if os.path.exists(mge_roi_pts_fn):
        # mge points
        mcbr_fn = re.sub('.npy','_withraw.npy', mge_roi_fn)
        mge_cell_bbox_roi = np.load(mcbr_fn)
        hcbrr_fn = re.sub('.npy','_withraw_resize.npy', hipr_roi_fn)
        hipr_cell_bbox_roi_resize = np.load(hcbrr_fn)
        mge_roi_pts = pd.read_csv(mge_roi_pts_fn)[['X','Y']].values
        fig, ax, cbar = ip.plot_image(mge_cell_bbox_roi)
        ax.plot(mge_roi_pts[:,0], mge_roi_pts[:,1], '.r', ms=mge_pts_size)
        plt.sca(ax)
        output_basename = re.sub('.npy','_register_points', mge_roi_fn)
        ip.save_png_pdf(output_basename)
        # Hipr points
        hipr_roi_pts = pd.read_csv(hipr_roi_pts_fn)[['X','Y']].values
        fig, ax, cbar = ip.plot_image(hipr_cell_bbox_roi_resize)
        ax.plot(hipr_roi_pts[:,0], hipr_roi_pts[:,1], '.r', ms=mge_pts_size)
        plt.sca(ax)
        output_basename = re.sub('.npy','_register_points', hipr_roi_fn)
        ip.save_png_pdf(output_basename)


# %% md

# Get hiprfish barcode rois and resize to mge

# %% codecell
np.sum(np.unique(hipr_bc) == 35)

for i in range(hipr_roi_props_df.shape[0]):
    row = hipr_roi_props_df.iloc[i,:]
    bbox = eval(row['bbox'])
    hipr_roi_fn = roi_npy_fns[sn_hipr][i]
    hipr_roi = np.load(hipr_roi_fn)
    hipr_bc_bbox = hipr_bc[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    hipr_roi_bbox = hipr_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    hipr_bc_bbox_roi = hipr_bc_bbox * (hipr_roi_bbox > 0)
    np.unique(hipr_bc_bbox_roi)

    res_mge = 0.02995
    res_hipr = 0.06748
    factor_resize = res_hipr / res_mge
    hipr_bc_bbox_roi_resize = resize(
                                    hipr_bc_bbox_roi, None, fx = factor_resize,
                                    fy = factor_resize, interpolation = INTER_NEAREST
                                    )
    output_fn = re.sub('.npy','_withbarcode.npy', hipr_roi_fn)
    np.save(output_fn, hipr_bc_bbox_roi_resize)
    print('Wrote:', output_fn)

# %% md

# Get raw spot rois and spot seg rois

# %% codecell
mge_spot = mge_raw[:,:,1]
mge_spot_tiff = Image.fromarray(mge_spot)
mge_spot_tiff.save(config['output_dir'] + '/mge_spot_raw.tif')
mge_spot_seg = ip.load_output_file(config, 'spot_seg_max_split_fmt', sn_mge_clip,
                                    cell_chan=0, spot_chan=1)
mge_roi_props_df = pd.read_csv(mge_roi_props_df_fn)
for i in range(mge_roi_props_df.shape[0]):
    # roi
    row = mge_roi_props_df.iloc[i,:]
    bbox = eval(row['bbox'])
    mge_roi_fn = roi_npy_fns[sn_mge][i]
    mge_roi = np.load(mge_roi_fn)
    mge_roi_bbox = mge_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    # spot seg
    mge_ss_bbox = mge_spot_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    mge_ss_bbox_roi = mge_ss_bbox * mge_roi_bbox
    output_fn = re.sub('.npy','_withspotseg.npy', mge_roi_fn)
    np.save(output_fn, mge_ss_bbox_roi)
    print('Wrote:', output_fn)
    # raw spot
    mge_sr_bbox = mge_spot[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    mge_sr_bbox_roi = mge_sr_bbox * (mge_roi_bbox > 0)
    output_fn = re.sub('.npy','_withrawspot.npy', mge_roi_fn)
    np.save(output_fn, mge_sr_bbox_roi)
    print('Wrote:', output_fn)


# %% md

# Now run warping

# %% codecell
mge_pts_weight = 0.8
for i in tqdm(range(mge_roi_props_df.shape[0])):
    # Get points
    mge_roi_fn = roi_npy_fns[sn_mge][i]
    hipr_roi_fn = roi_npy_fns[sn_hipr][i]
    mge_roi_pts_fn = re.sub('.npy','_withraw_pts.csv',mge_roi_fn)
    hipr_roi_pts_fn = re.sub('.npy','_withraw_resize_pts.csv',hipr_roi_fn)
    if os.path.exists(mge_roi_pts_fn):
        mge_roi_pts = pd.read_csv(mge_roi_pts_fn)[['X','Y']].values
        hipr_roi_pts = pd.read_csv(hipr_roi_pts_fn)[['X','Y']].values
        result_points = weighted_average_points(mge_roi_pts, hipr_roi_pts, mge_pts_weight)
        # Get buffers for final warped images size
        mcbr_fn = re.sub('.npy','_withraw.npy', mge_roi_fn)
        mge_cell_bbox_roi = np.load(mcbr_fn)
        hcbrr_fn = re.sub('.npy','_withraw_resize.npy', hipr_roi_fn)
        hipr_cell_bbox_roi_resize = np.load(hcbrr_fn)
        hbbrr_fn = re.sub('.npy','_withbarcode.npy', hipr_roi_fn)
        hipr_bc_bbox_roi_resize = np.load(hbbrr_fn)
        msbr_fn = re.sub('.npy','_withspotseg.npy', mge_roi_fn)
        mge_ss_bbox_roi = np.load(msbr_fn)
        mge_sr_bbox_roi_fn = re.sub('.npy','_withrawspot.npy', mge_roi_fn)
        mge_sr_bbox_roi = np.load(mge_sr_bbox_roi_fn)
        ims = [hipr_bc_bbox_roi_resize, mge_ss_bbox_roi]
        size=[0,0]
        for i in range(2):
            shps = [im.shape[i] for im in ims]
            size[i] = np.min(shps) + 2*(np.max(shps) - np.min(shps))
        print(size, hipr_bc_bbox_roi_resize.shape, mge_ss_bbox_roi.shape)
        # mge_cell warp
        mcbr_wrp = warp_image(
                        mge_cell_bbox_roi, mge_roi_pts,
                        result_points, size, dtype=np.int32
                        )
        output_fn = re.sub('.npy','_withraw_warped.npy', mge_roi_fn)
        np.save(output_fn, mcbr_wrp)
        # hipr raw warp
        hcbrr_warp = warp_image(
                        hipr_cell_bbox_roi_resize, hipr_roi_pts,
                        result_points, size, dtype=np.int32
                        )
        output_fn = re.sub('.npy','_withraw_resize_warped.npy', mge_roi_fn)
        np.save(output_fn, hcbrr_warp)
        # barcode warp
        hbbrr_wrp = warp_image(
                        hipr_bc_bbox_roi_resize, hipr_roi_pts,
                        result_points, size, dtype=np.int32,
                        interpolation='nearest'
                        )
        output_fn = re.sub('.npy','_withbarcode_warped.npy', mge_roi_fn)
        np.save(output_fn, hbbrr_wrp)
        # Spot set warp
        msbr_wrp = warp_image(
                    mge_ss_bbox_roi, mge_roi_pts, result_points, size,
                    dtype=np.int32, interpolation='nearest'
                    )
        output_fn = re.sub('.npy','_withspotseg_warped.npy', mge_roi_fn)
        np.save(output_fn, msbr_wrp)
        print('Wrote:', output_fn)
        # Spot raw warp
        mge_sr_bbox_roi_wrp = warp_image(mge_sr_bbox_roi, mge_roi_pts, result_points, size,
                                dtype=np.float64)
        output_fn = re.sub('.npy','_withrawspot_warped.npy', mge_roi_fn)
        np.save(output_fn, mge_sr_bbox_roi_wrp)
        print('Wrote:', output_fn)

# %% md

# save overlay plots of raw images

# %% codecell
for i in tqdm(range(mge_roi_props_df.shape[0])):
    # Get points
    mge_roi_fn = roi_npy_fns[sn_mge][i]
    hipr_roi_fn = roi_npy_fns[sn_hipr][i]
    mge_roi_pts_fn = re.sub('.npy','_withraw_pts.csv',mge_roi_fn)
    hipr_roi_pts_fn = re.sub('.npy','_withraw_resize_pts.csv',hipr_roi_fn)
    if os.path.exists(mge_roi_pts_fn):
        mcbr_fn = re.sub('.npy','_withraw.npy', mge_roi_fn)
        mge_cell_bbox_roi = np.load(mcbr_fn)
        hcbrr_fn = re.sub('.npy','_withraw_resize.npy', hipr_roi_fn)
        hipr_cell_bbox_roi_resize = np.load(hcbrr_fn)
        mcbr_wrp_fn = re.sub('.npy','_withraw_warped.npy', mge_roi_fn)
        mcbr_wrp = np.load(mcbr_wrp_fn)
        hcbrr_warp_fn = re.sub('.npy','_withraw_resize_warped.npy', mge_roi_fn)
        hcbrr_warp = np.load(hcbrr_warp_fn)
        shape = [0,0]
        ims = [mge_cell_bbox_roi, hipr_cell_bbox_roi_resize]
        for i in range(2):
            shps = [im.shape[i] for im in ims]
            shape[i] = np.max(shps)
        mcbr = np.zeros(shape)
        shp = mge_cell_bbox_roi.shape
        mcbr[:shp[0],:shp[1]] = mge_cell_bbox_roi
        mcbr_norm = ip.zero_one_thresholding(mcbr, clims=(0,0.7))
        hcbrr = np.zeros(shape)
        shp = hipr_cell_bbox_roi_resize.shape
        hcbrr[:shp[0],:shp[1]] = hipr_cell_bbox_roi_resize
        hcbrr_norm = ip.zero_one_thresholding(hcbrr, clims=(0,1))
        overlay_shift = np.dstack([mcbr_norm, hcbrr_norm, np.zeros(hcbrr_norm.shape)])
        # morph images
        roi_mge_wrp_norm = ip.zero_one_thresholding(mcbr_wrp, clims=(0,0.7))
        roi_hipr_wrp_norm = ip.zero_one_thresholding(hcbrr_warp, clims=(0,1))
        overlay_warp = np.dstack([roi_mge_wrp_norm, roi_hipr_wrp_norm, np.zeros(roi_hipr_wrp_norm.shape)])

        image_list = [
                    mcbr_norm, hcbrr_norm, overlay_shift,
                    roi_mge_wrp_norm, roi_hipr_wrp_norm, overlay_warp
                    ]
        cmaps = ['inferno','inferno','']*2
        ip.subplot_square_images(image_list,(2,3),cmaps=cmaps)
        bn = re.sub('.npy','',os.path.basename(mge_roi_fn))
        output_basename = config['output_dir'] + '/roi_masks/combined_outputs/' + bn
        ip.save_png_pdf(output_basename)



# %% md

# Save overlay plots of segmentations

# %% codecell
im_inches=10
marker='.'
spot_col=np.array([0,1.,0,0.5])
marker_size=2000
linewidths=5
mge_roi_props_df.columns
for mge_roi_fn, hipr_roi_fn in tqdm(zip(roi_npy_fns[sn_mge], roi_npy_fns[sn_hipr])):
    # Get warps
    hbbrr_wrp_fn = re.sub('.npy','_withbarcode_warped.npy', mge_roi_fn)
    mssbr_wrp_fn = re.sub('.npy','_withspotseg_warped.npy', mge_roi_fn)
    msrbr_wrp_fn = re.sub('.npy','_withrawspot_warped.npy', mge_roi_fn)
    if os.path.exists(hbbrr_wrp_fn):
        hbbrr_wrp = np.load(hbbrr_wrp_fn)
        # Save warped ident npy
        hbbrr_wrp_tab20 = ip.color_bc_image(hbbrr_wrp, barcodes, cols_t20_ordered_0)
        output_basename = re.sub('.npy','_warped_ident.npy', hipr_roi_fn)
        np.save(output_basename, hbbrr_wrp_tab20)
        print('Wrote:', output_basename)
        # Save warped ident pdf
        fig, ax, cbar = ip.plot_image(hbbrr_wrp_tab20, im_inches=im_inches)
        output_basename = re.sub('.npy','_warped_ident', hipr_roi_fn)
        ip.save_png_pdf(output_basename)
        print('Wrote:', output_basename)
        plt.close()
        # Save warped spot raw
        msrbr_wrp = np.load(msrbr_wrp_fn)
        fig, ax, cbar = ip.plot_image(msrbr_wrp, im_inches=im_inches)
        output_basename = re.sub('.npy','_withrawspot_warped', mge_roi_fn)
        ip.save_png_pdf(output_basename)
        print('Wrote:', output_basename)
        plt.close()
        # Save spot overlay on warped ident
        mssbr_wrp = np.load(mssbr_wrp_fn)
        mssbr_wrp_props = sf.measure_regionprops(
                            mssbr_wrp, raw=msrbr_wrp
                            )  # spot seg properties
        spot_int = mssbr_wrp_props.max_intensity.values
        spot_int /= np.max(spot_int)
        marker_size_arr = marker_size * spot_int
        fig, ax, cbar = ip.plot_image(hbbrr_wrp_tab20, im_inches=im_inches)
        sc = mssbr_wrp_props.centroid
        sc = np.array([[i for i in k] for k in sc])
        ax.scatter(sc[:,1], sc[:,0], marker=marker, s=marker_size_arr, color=spot_col, linewidths=linewidths, edgecolors='none')
        bn = re.sub('.npy','',os.path.basename(hipr_roi_fn))
        output_basename = config['output_dir'] + '/roi_masks/combined_outputs/' + bn + '_ident_warped_spot_overlay'
        ip.save_png_pdf(output_basename)
        print('Wrote:', output_basename)

# %% md

# =============================================================================
# ## Spot host colocalization
# =============================================================================

# Get colocalization of spot with host

# %% codecell
hipr_bc_bbox_roi_resize
from sklearn.neighbors import NearestNeighbors

spot_props_roi = pd.DataFrame([])
for i in tqdm(range(mge_roi_props_df.shape[0])):
    # Get warps
    mge_roi_fn = roi_npy_fns[sn_mge][i]
    hipr_roi_fn = roi_npy_fns[sn_hipr][i]
    hbbrr_wrp_fn = re.sub('.npy','_withbarcode_warped.npy', mge_roi_fn)
    mssbr_wrp_fn = re.sub('.npy','_withspotseg_warped.npy', mge_roi_fn)
    msrbr_wrp_fn = re.sub('.npy','_withrawspot_warped.npy', mge_roi_fn)
    if os.path.exists(hbbrr_wrp_fn):
        hbbrr_wrp = np.load(hbbrr_wrp_fn)
        hbbrr_wrp_props = sf.measure_regionprops(
                            sf.label(hbbrr_wrp > 0), raw=hbbrr_wrp
                            )
        mssbr_wrp = np.load(mssbr_wrp_fn)
        msrbr_wrp = np.load(msrbr_wrp_fn)
        mssbr_wrp_props = sf.measure_regionprops(
                            mssbr_wrp, raw=msrbr_wrp
                            )
        host_list = []
        cells = [[y,x] for y, x in hbbrr_wrp_props.centroid.values]
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(cells)
        for y, x in mssbr_wrp_props.centroid.values:
            host = hbbrr_wrp[int(y), int(x)]
            if host == 0:
                distances, indices = nbrs.kneighbors([[y,x]])
                host = hbbrr_wrp_props.loc[indices[0][0], 'max_intensity']
            host_list.append(host)
        mssbr_wrp_props['host_barcode'] = host_list
        mssbr_wrp_props['roi'] = roi_npy_labels[sn_mge][i]
        spot_props_roi = spot_props_roi.append(mssbr_wrp_props)
spot_props_roi.to_csv(config['output_dir'] + '/spot_props_roi.csv')


# %% md

# Plot species spot count

# %% codecell
host_count = spot_props_roi['host_barcode'].value_counts()
color=[]
label=[]
for h in host_count.index:
    if h != 0:
        row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
        col = row['tab20']
        lab = row['sci_name']
        if col.shape[0] > 0:
            color.append(col.values[0])
            label.append(lab.values[0])
        else:
            color.append((0,0,1))
            label.append('weird...')
    else:
        color.append((0,0,0))
        label.append('no host')
ax = host_count.plot.bar(color=color, tick_label=label)
ax.set_xticklabels(label, rotation=90)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_all'
ip.save_png_pdf(output_basename)
# ax.set_ylim(0,100)

# %% md

# Same plot but grouped by roi

# %% codecell
def plot_hosts(df, taxon_color_lookup):
    host_count = df['host_barcode'].value_counts()
    color=[]
    label=[]
    for h in host_count.index:
        if h != 0:
            row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
            col = row['tab20']
            lab = row['sci_name']
            if col.shape[0] > 0:
                color.append(col.values[0])
                label.append(lab.values[0])
            else:
                color.append((0,0,1))
                label.append('weird...')
        else:
            color.append((0,0,0))
            label.append('no host')
    fig, ax = ip.general_plot()
    x = np.arange(host_count.shape[0])
    heights = host_count.values
    ax.bar(x, heights, color=color, tick_label=label)
    ax.set_xticklabels(label, rotation=90)
    return ax

for roi in spot_props_roi['roi'].unique():
    df = spot_props_roi[spot_props_roi['roi'] == roi]
    plot_hosts(df, taxon_color_lookup)
    output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_roi_' + str(roi)
    ip.save_png_pdf(output_basename)
    # plt.title('ROI: ' + roi)
    plt.show()
    plt.close()

# %% md

# Plot spot intensities for each association

# %% codecell
ylim=0.06

spot_props_roi = spot_props_roi.sort_values(by='max_intensity')
color = []
for h in host_count.index:
    fig, ax = ip.general_plot()
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    df = spot_props_roi[spot_props_roi['host_barcode'] == h]
    if h != 0:
        if row.shape[0] > 0:
            col = row['tab20'].values[0]
            lab = row['sci_name'].values[0]
        else:
            col = (0,0,0)
            lab = 'weird...'
    else:
        col = (0,0,0)
        lab = 'No Host'
    x = np.arange(df.shape[0])
    heights = df['max_intensity'].values
    ax.bar(x, heights, color=col)
    ax.set_ylim([0,ylim])
    output_basename = config['output_dir'] + '/spot_host_association/spot_association_intensity_genus_' + lab
    ip.save_png_pdf(output_basename)
    # plt.title(lab)
    # plt.ylabel('Spot intensity')
    # plt.xlabel('Ordered spots')
    plt.show()
    plt.close()

# %% md

# Plot spot intensity colored for all associations

# %% codecell
spot_props_roi = spot_props_roi.sort_values(by='max_intensity')
x = np.arange(spot_props_roi.shape[0])
height = spot_props_roi['max_intensity'].values
color = []
for h in spot_props_roi['host_barcode'].unique():
    if h != 0:
        row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
        col = row['tab20']
        lab = row['sci_name']
        if col.shape[0] > 0:
            color.append(col.values[0])
            label.append(lab.values[0])
        else:
            color.append((0,0,1))
            label.append('weird...')
    else:
        color.append((0,0,0))
        label.append('no host')
plt.bar(x, height, color=color)
# plt.ylabel('Spot intensity')
# plt.xlabel('Ordered spots')
output_basename = config['output_dir'] + '/spot_host_association/spot_intensity_all'
ip.save_png_pdf(output_basename)

# %% md

# Pick filter value

# %% codecell
spot_props_roi.iloc[475,:]['max_intensity']

# %% codecell
filt_maxint=0.01764

x = np.arange(spot_props_roi.shape[0])
height = spot_props_roi['max_intensity'].values
fig, ax = ip.general_plot()
ax.bar(x, height, width=1)
ax.plot([0,spot_props_roi.shape[0]], [filt_maxint]*2, '-k')
output_basename = config['output_dir'] + '/spot_host_association/spot_intensity_all_threshold'
ip.save_png_pdf(output_basename)
# plt.ylabel('Spot intensity')
# plt.xlabel('Ordered spots')
spr_filt = spot_props_roi[spot_props_roi['max_intensity'] > filt_maxint]
print(spot_props_roi.shape[0])
print(spr_filt.shape[0])

# %% md

# Re-plot species count with filtering

# %% codecell
host_count = spr_filt['host_barcode'].value_counts()
color=[]
label=[]
for h in host_count.index:
    if h != 0:
        row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
        col = row['tab20']
        lab = row['sci_name']
        if col.shape[0] > 0:
            color.append(col.values[0])
            label.append(lab.values[0])
        else:
            color.append((0,0,1))
            label.append('weird...')
    else:
        color.append((0,0,0))
        label.append('no host')
ax = host_count.plot.bar(color=color, tick_label=label)
ax.set_xticklabels(label, rotation=90)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_all_filtintensity'
ip.save_png_pdf(output_basename)
# ax.set_ylim(0,100)

# %% md

# Same plot but grouped by roi

# %% codecell
ylim=10

def plot_hosts(df, taxon_color_lookup):
    host_count = df['host_barcode'].value_counts()
    color=[]
    label=[]
    for h in host_count.index:
        if h != 0:
            row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
            col = row['tab20']
            lab = row['sci_name']
            if col.shape[0] > 0:
                color.append(col.values[0])
                label.append(lab.values[0])
            else:
                color.append((0,0,1))
                label.append('weird...')
        else:
            color.append((0,0,0))
            label.append('no host')
    fig, ax = ip.general_plot()
    x = np.arange(host_count.shape[0])
    heights = host_count.values
    ax.bar(x, heights, color=color, tick_label=label)
    ax.set_xticklabels(label, rotation=90)
    return ax

for roi in spr_filt['roi'].unique():
    df = spr_filt[spr_filt['roi'] == roi]
    ax = plot_hosts(df, taxon_color_lookup)
    ax.set_ylim([0,ylim])
    output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_filtintensity_roi_' + str(roi)
    ip.save_png_pdf(output_basename)
    # plt.title('ROI: ' + roi)
    plt.show()
    plt.close()

# %% md

# Plot spot intensities for each association after filtering

# %% codecell
ylim=0.06

color = []
for h in host_count.index:
    fig, ax = ip.general_plot()
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    df = spr_filt[spr_filt['host_barcode'] == h]
    if h != 0:
        if row.shape[0] > 0:
            col = row['tab20'].values[0]
            lab = row['sci_name'].values[0]
        else:
            col = (0,0,0)
            lab = 'weird...'
    else:
        col = (0,0,0)
        lab = 'No Host'
    x = np.arange(df.shape[0])
    heights = df['max_intensity'].values
    ax.bar(x, heights, color=col)
    ax.set_ylim([0,ylim])
    output_basename = config['output_dir'] + '/spot_host_association/spot_association_intensity_filtintensity_genus_' + lab
    ip.save_png_pdf(output_basename)
    # plt.title(lab)
    plt.ylabel('Spot intensity')
    plt.xlabel('Ordered spots')
    plt.show()
    plt.close()

# %% md

# Plot global spot associations but with variance between rois.

# Normalize by number of taxon in roi

# %% codecell
markersize = 5
marker = 'o'
ylim = 0.4
dims = (5,5)

host_count = spot_props_roi['host_barcode'].value_counts()
rois = spot_props_roi['roi'].unique()
color=[]
label=[]
fig, ax = ip.general_plot(dims=dims)
from random import seed, random
seed(42)
labels = []
tax_counts = []
cell_counts = []
for i, h in enumerate(host_count.index):
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    col = row['tab20']
    lab = row['sci_name']
    if col.shape[0] > 0:
        color = col.values[0]
        label = lab.values[0]
        labels.append(label)
    else:
        color = (0,0,1)
        label = 'weird...'
        labels.append(label)
    values = []
    for roi in rois:
        hipr_df = hipr_seg_roi_props[hipr_seg_roi_props['roi'] == roi]
        spot_df = spot_props_roi[spot_props_roi['roi'] == roi]
        cell_count = hipr_df.shape[0]
        tax_count = hipr_df[hipr_df['max_intensity'] == h].shape[0]
        tax_counts.append(tax_count)
        cell_counts.append(cell_count)
        if tax_count > 1:
            assoc_count = spot_df[spot_df['host_barcode'] == h].shape[0]
            value = assoc_count / (tax_count + 1e-15)
            values.append(value)
            if value > 0.4:
                print(value, assoc_count, tax_count, cell_count, label)
            r = random()*0.5 - 0.25
            ax.plot(i + r, value, marker=marker, color=color, markersize=markersize)
    ax.plot([i-0.25, i+0.25], [np.median(values)]*2, color='k')
ax.set_xticks(np.arange(i+1))
ax.set_xticklabels(labels, rotation=90)
ax.set_ylim(0,ylim)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_all_rois_normtax'
ip.save_png_pdf(output_basename)

# %% md

# Global spot association normalize by taxon count filter dim spots

# %% codecell
markersize = 7
marker = 'o'
ylim = 0.2
dims = (5,4)

host_count = spr_filt['host_barcode'].value_counts()
rois = spr_filt['roi'].unique()
color=[]
label=[]
fig, ax = ip.general_plot(dims=dims)
from random import seed, random
seed(42)
labels = []
tax_counts = []
cell_counts = []
for i, h in enumerate(host_count.index):
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    col = row['tab20']
    lab = row['sci_name']
    if col.shape[0] > 0:
        color = col.values[0]
        label = lab.values[0]
        labels.append(label)
    else:
        color = (0,0,1)
        label = 'weird...'
        labels.append(label)
    values = []
    for roi in rois:
        hipr_df = hipr_seg_roi_props[hipr_seg_roi_props['roi'] == roi]
        spot_df = spr_filt[spr_filt['roi'] == roi]
        cell_count = hipr_df.shape[0]
        tax_count = hipr_df[hipr_df['max_intensity'] == h].shape[0]
        tax_counts.append(tax_count)
        cell_counts.append(cell_count)
        if tax_count > 5:
            assoc_count = spot_df[spot_df['host_barcode'] == h].shape[0]
            value = assoc_count / (tax_count + 1e-15)
            if value > ylim:
                print(value, assoc_count, tax_count, cell_count, label)
            if value < ylim:
                values.append(value)
            r = random()*0.5 - 0.25
            ax.plot(i + r, value, marker=marker, color=color, markersize=markersize)
    ax.plot([i-0.25, i+0.25], [np.mean(values)]*2, color='k')
ax.set_xticks(np.arange(i+1))
ax.set_xticklabels(labels, rotation=90)
ax.set_ylim(0,ylim)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_all_rois_normtax_filtint'
ip.save_png_pdf(output_basename)

# %% md

# Plot global spot associations but with variance between rois.

# Normalize by number of taxon in roi and number of cells in roi

# %% codecell
markersize = 5
marker = 'o'
ylim = 0.0006
dims = (5,5)

host_count = spot_props_roi['host_barcode'].value_counts()
rois = spot_props_roi['roi'].unique()
color=[]
label=[]
fig, ax = ip.general_plot(dims=dims)
from random import seed, random
seed(42)
labels = []
tax_counts = []
cell_counts = []
for i, h in enumerate(host_count.index):
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    col = row['tab20']
    lab = row['sci_name']
    if col.shape[0] > 0:
        color = col.values[0]
        label = lab.values[0]
        labels.append(label)
    else:
        color = (0,0,1)
        label = 'weird...'
        labels.append(label)
    values = []
    for roi in rois:
        hipr_df = hipr_seg_roi_props[hipr_seg_roi_props['roi'] == roi]
        spot_df = spot_props_roi[spot_props_roi['roi'] == roi]
        cell_count = hipr_df.shape[0]
        tax_count = hipr_df[hipr_df['max_intensity'] == h].shape[0]
        tax_counts.append(tax_count)
        cell_counts.append(cell_count)
        if tax_count > 1:
            assoc_count = spot_df[spot_df['host_barcode'] == h].shape[0]
            value = assoc_count / (tax_count + 1e-15) / (cell_count + 1e-15)
            values.append(value)
            if value > 0.0005:
                print(value, assoc_count, tax_count, cell_count, label)
            r = random()*0.5 - 0.25
            ax.plot(i + r, value, marker=marker, color=color, markersize=markersize)
    ax.plot([i-0.25, i+0.25], [np.median(values)]*2, color='k')
ax.set_xticks(np.arange(i+1))
ax.set_xticklabels(labels, rotation=90)
ax.set_ylim(0,ylim)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_all_rois_normboth'
ip.save_png_pdf(output_basename)

# %% md

# Plot scatter of fraction of taxon with spots vs fraction of cells that are taxon

# Define quadrants

# %% codecell
markersize = 5
marker = 'o'
xlim = 0.6
ylim = 0.6
dims = (5,5)
ft = 12
qx = 0.05
qy = 0.05

host_count = spot_props_roi['host_barcode'].value_counts()
rois = spot_props_roi['roi'].unique()
color=[]
label=[]
fig, ax = ip.general_plot(dims=dims)
# plot quadrant lines
ax.plot([qx,qx],[0,ylim], 'k')
ax.plot([0,xlim],[qy,qy], 'k')
from random import seed, random
seed(42)
labels = []
tax_counts = []
cell_counts = []
for i, h in enumerate(host_count.index):
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    col = row['tab20']
    lab = row['sci_name']
    if col.shape[0] > 0:
        color = col.values[0]
        label = lab.values[0]
        labels.append(label)
    else:
        color = (0,0,1)
        label = 'weird...'
        labels.append(label)
    values = []
    for j, roi in enumerate(rois):
        hipr_df = hipr_seg_roi_props[hipr_seg_roi_props['roi'] == roi]
        spot_df = spot_props_roi[spot_props_roi['roi'] == roi]
        cell_count = hipr_df.shape[0]
        tax_count = hipr_df[hipr_df['max_intensity'] == h].shape[0]
        tax_counts.append(tax_count)
        cell_counts.append(cell_count)
        if tax_count > 0:
            assoc_count = spot_df[spot_df['host_barcode'] == h].shape[0]
            value = assoc_count / (tax_count + 1e-15) / (cell_count + 1e-15)
            values.append(value)

            # color = color if (tc > tc_lim)
            r = random()*0.5 - 0.25
            if j == 0:
                ax.plot(
                    tax_count/cell_count, assoc_count/tax_count,
                    marker=marker,
                    color=color,
                    markersize=markersize,
                    label=label
                    )
            else:
                ax.plot(
                    tax_count/cell_count, assoc_count/tax_count,
                    marker=marker,
                    color=color,
                    markersize=markersize,
                    label='_nolegend_'
                    )
ax.legend()
ax.set_xlabel('n taxa / n cells in ROI')
ax.set_ylabel('n mges in taxa / n taxa in ROI')
ax.set_ylim(0,ylim)
ax.set_xlim(0,xlim)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_all_rois_norm_scatter'
ip.save_png_pdf(output_basename)

# %% md

# Same plot with intensity filtering

# %% codecell
markersize = 7
marker = 'o'
xlim = 0.6
ylim = 0.6
dims = (5,5)
ft = 18
qx = 0.05
qy = 0.025

host_count = spr_filt['host_barcode'].value_counts()
rois = spr_filt['roi'].unique()
color=[]
label=[]
fig, ax = ip.general_plot(dims=dims, ft=ft)
# plot quadrant lines
# ax.plot([qx,qx],[0,ylim], 'k')
# ax.plot([0,xlim],[qy,qy], 'k')
from random import seed, random
seed(42)
labels = []
tax_counts = []
cell_counts = []
for i, h in enumerate(host_count.index):
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    col = row['tab20']
    lab = row['sci_name']
    if col.shape[0] > 0:
        color = col.values[0]
        label = lab.values[0]
        labels.append(label)
    else:
        color = (0,0,1)
        label = 'weird...'
        labels.append(label)
    values = []
    for j, roi in enumerate(rois):
        hipr_df = hipr_seg_roi_props[hipr_seg_roi_props['roi'] == roi]
        spot_df = spr_filt[spr_filt['roi'] == roi]
        cell_count = hipr_df.shape[0]
        tax_count = hipr_df[hipr_df['max_intensity'] == h].shape[0]
        tax_counts.append(tax_count)
        cell_counts.append(cell_count)
        if tax_count > 0:
            assoc_count = spot_df[spot_df['host_barcode'] == h].shape[0]
            value = assoc_count / (tax_count + 1e-15) / (cell_count + 1e-15)
            values.append(value)

            # color = color if (tc > tc_lim)
            r = random()*0.5 - 0.25
            if j == 0:
                ax.plot(
                    tax_count/cell_count, assoc_count/tax_count,
                    marker=marker,
                    color=color,
                    markersize=markersize,
                    label=label
                    )
            else:
                ax.plot(
                    tax_count/cell_count, assoc_count/tax_count,
                    marker=marker,
                    color=color,
                    markersize=markersize,
                    label='_nolegend_'
                    )
ax.legend()
ax.set_xlabel('n taxa / n cells in ROI')
ax.set_ylabel('n mges in taxa / n taxa in ROI')
ax.set_ylim(0,ylim)
ax.set_xlim(0,xlim)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_all_rois_norm_scatter_filtint'
ip.save_png_pdf(output_basename)

# %% md

# same plot as immediately before, but with rois combined

# %% codecell
markersize = 10
marker = 'o'
xlim = 0.6
ylim = 0.1
dims = (5,5)
ft = 18
qx = 0.05
qy = 0.025

host_count = spr_filt['host_barcode'].value_counts()
rois = spr_filt['roi'].unique()
color=[]
label=[]
fig, ax = ip.general_plot(dims=dims, ft=ft)
# plot quadrant lines
# ax.plot([qx,qx],[0,ylim], 'k')
# ax.plot([0,xlim],[qy,qy], 'k')
from random import seed, random
seed(42)
labels = []
for i, h in enumerate(host_count.index):
    row = taxon_color_lookup.loc[taxon_color_lookup.code_b10 == h,:]
    col = row['tab20']
    lab = row['sci_name']
    if col.shape[0] > 0:
        color = col.values[0]
        label = lab.values[0]
        labels.append(label)
    else:
        color = (0,0,1)
        label = 'weird...'
        labels.append(label)
    tax_counts = 0
    cell_counts = 0
    assoc_counts = 0
    for j, roi in enumerate(rois):
        hipr_df = hipr_seg_roi_props[hipr_seg_roi_props['roi'] == roi]
        spot_df = spr_filt[spr_filt['roi'] == roi]
        cell_count = hipr_df.shape[0]
        tax_count = hipr_df[hipr_df['max_intensity'] == h].shape[0]
        assoc_count = spot_df[spot_df['host_barcode'] == h].shape[0]
        tax_counts += tax_count
        cell_counts += cell_count
        assoc_counts += assoc_count
    if label=='Rothia':
        print(assoc_counts, tax_counts)
    ax.plot(
        tax_counts/cell_counts, assoc_counts/tax_counts,
        marker=marker,
        color=color,
        markersize=markersize,
        label=label
        )
ax.legend()
ax.set_xlabel('n taxa / n cells in ROI')
ax.set_ylabel('n mges in taxa / n taxa in ROI')
ax.set_ylim(0,ylim)
# ax.set_xlim(0,xlim)
output_basename = config['output_dir'] + '/spot_host_association/species_spot_count_combine_rois_norm_scatter_filtint'
ip.save_png_pdf(output_basename)


# %% md

# Group points in plot by quadrant, define an xlim and ylim to create quadrants

# Plot intensity values for subsetted points

# Plot

# %% md

# =============================================================================
# ## Pair correlation function
# =============================================================================

# ### Instead of warping to points halfway between mge and hipr, warp mge spots to hipr coordinates

# get roi props in terms of rescaled hiprfish image

# %% codecell
# Resize
hipr_registered_fn = config['output_dir'] + '/20222_03_19_plaqueplasmidamr_sample_hs_probe_mefe_fov_tile1_round_2_mode_spec_registered.npy'
hipr_registered = np.load(hipr_registered_fn)
hipr_max = np.max(hipr_registered, axis=2)
hipr_max_resize = resize(
                    hipr_max, None, fx = factor_resize,
                    fy = factor_resize, interpolation = INTER_NEAREST
                    )
hipr_max_resize.shape

# %% codecell
# Get props
hipr_roi_res_props_list = []
# Get region properties
for i, fn in tqdm(enumerate(roi_npy_fns[sn_hipr])):
    roi = np.load(fn)
    roi_resize = resize(
                    roi, None, fx = factor_resize,
                    fy = factor_resize, interpolation = INTER_NEAREST
                    )
    props_ = sf.measure_regionprops(roi_resize)
    hipr_roi_res_props_list.append(props_)

hipr_roi_res_props_df = pd.concat(hipr_roi_res_props_list)
hipr_roi_res_props_df_fn = config['output_dir'] + '/roi_masks/' + sn_hipr + '_roi_res_props.csv'
hipr_roi_res_props_df.to_csv(hipr_roi_res_props_df_fn)


# %% md

# Place hipr points in context of full image

# %% codecell
mge_spot_seg_wrp = np.zeros(hipr_max_resize.shape)
mge_spot_raw_wrp = np.zeros(hipr_max_resize.shape)

for i, (mge_roi_fn, hipr_roi_fn) in tqdm(enumerate(zip(
                                        roi_npy_fns[sn_mge],
                                        roi_npy_fns[sn_hipr]
                                        ))):
    mge_roi_pts_fn = re.sub('.npy','_withraw_pts.csv',mge_roi_fn)
    hipr_roi_pts_fn = re.sub('.npy','_withraw_resize_pts.csv',hipr_roi_fn)
    if os.path.exists(mge_roi_pts_fn):
        # Get mge seg and raw rois and sister points
        mssbr_fn = re.sub('.npy','_withspotseg.npy', mge_roi_fn)
        mge_ss_bbox_roi = np.load(mssbr_fn)
        mge_sr_bbox_roi_fn = re.sub('.npy','_withrawspot.npy', mge_roi_fn)
        mge_sr_bbox_roi = np.load(mge_sr_bbox_roi_fn)
        mge_roi_pts = pd.read_csv(mge_roi_pts_fn)[['X','Y']].values
        # Get hipr sister points and roi coords
        hipr_roi_pts = pd.read_csv(hipr_roi_pts_fn)[['X','Y']].values
        hipr_roi_bbox = hipr_roi_res_props_df.iloc[i,:]['bbox']
        hipr_roi_pts[:,0] += hipr_roi_bbox[1]
        hipr_roi_pts[:,1] += hipr_roi_bbox[0]
        hipr_roi_pts = hipr_roi_pts.astype(np.int64)

        # Warp mge spot raw
        msrr_wrp = warp_image(
                    mge_sr_bbox_roi, mge_roi_pts, hipr_roi_pts,
                    hipr_max_resize.shape, dtype=np.float64
                    )
        np.max(hipr_roi_pts, axis=0)
        # warp mge spot seg
        mssr_wrp = warp_image(
                    mge_ss_bbox_roi, mge_roi_pts, hipr_roi_pts,
                    hipr_max_resize.shape, dtype=np.int64,
                    interpolation='nearest'
                    )
        # Combine warped spot seg and spot raw into full image
        mge_spot_seg_wrp += mssr_wrp
        mge_spot_raw_wrp += msrr_wrp


mssw_bn = (config['output_dir'] +
                '/roi_masks/combined_outputs/'
                + sn_hipr + '_spot_seg_rois_warped'
                )
msrw_bn = (config['output_dir'] +
                '/roi_masks/combined_outputs/'
                + sn_hipr + '_spot_raw_rois_warped'
                )
# Save np files
mge_spot_seg_wrp = mge_spot_seg_wrp.astype(np.int64)
np.save(mssw_bn + '.npy', mge_spot_seg_wrp)
np.save(msrw_bn + '.npy', mge_spot_raw_wrp)
# Save plots
ip.plot_image(mge_spot_raw_wrp, cmap='inferno')
ip.save_png_pdf(msrw_bn)
plt.show()
plt.close()
ip.plot_image(ip.seg2rgb(mge_spot_seg_wrp))
ip.save_png_pdf(mssw_bn)
plt.show()
plt.close()

# %% md

# Get spot warped spot props full

# %% codecell
mge_spot_seg_warp_props = sf.measure_regionprops(
                        mge_spot_seg_wrp,
                        raw=mge_spot_raw_wrp
                        )
msswp_fn = (config['output_dir'] +
                '/roi_masks/combined_outputs/'
                + sn_hipr + '_spot_seg_rois_warped_props.csv'
                )
mge_spot_seg_warp_props.to_csv(msswp_fn)

# %% md

# Plot mge spots on full image

# %% codecell
# resized barcode image
hipr_bc_resize = resize(
                    hipr_bc, None, fx = factor_resize,
                    fy = factor_resize, interpolation = INTER_NEAREST
                    )
# Recolor bc image
[bc for bc in np.unique(hipr_bc_resize) if bc not in taxon_color_lookup.code_b10.values]
barcodes_withna = np.concatenate([barcodes,np.array([35])])
cols_t20_ordered_0_withna = np.concatenate([cols_t20_ordered_0, [[0,0,1]]]).tolist()
hbbrr_bc_resize_tab20 = ip.color_bc_image(
                            hipr_bc_resize.astype(np.int64),
                            barcodes_withna,
                            cols_t20_ordered_0_withna
                            )
# %% codecell
# PLot
fig, ax, cbar = ip.plot_image(hbbrr_bc_resize_tab20, scalebar_resolution=res_mge)
ax.scatter(ref_pts[0], ref_pts[1], color=(0,1,0), )
im_inches=10
marker='.'
spot_col=np.array([0,1.,0,0.5])
marker_size=100
linewidths=5
ref_pts = [list(c) for c in mge_spot_seg_warp_props['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
ref_pts_arr = np.array(ref_pts)
spot_int = mge_spot_seg_warp_props.max_intensity.values
spot_int /= np.max(spot_int)
marker_size_arr = marker_size * spot_int
sc = mssbr_wrp_props.centroid
sc = np.array([[i for i in k] for k in sc])
ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
            marker=marker, s=marker_size_arr, color=spot_col,
            linewidths=linewidths, edgecolors='none'
            )
output_basename = config['output_dir'] + '/spot_host_association/hipr_ident_spot_overlay'
ip.save_png_pdf(output_basename)


# %% md

# ### Calculate radial density functions

# Get coordinates of all taxa of interest

# %% codecell

target_taxon='Fretibacterium'  # Taxon of interest
tax_code = taxon_color_lookup.loc[
            taxon_color_lookup['sci_name'] == target_taxon,
            'code'
            ].values[0]

hipr_cell_info = sf.measure_regionprops(
                    sf.label(hipr_bc_resize > 0),
                    raw=hipr_bc_resize
                    )
hipr_cell_info['cell_barcode_b10'] = hipr_cell_info['max_intensity'].astype(int)
bcs_b10 = hipr_cell_info.cell_barcode_b10.values
bcs_b2 = [int(bin(cb)[2:]) for cb in bcs_b10]
hipr_cell_info['cell_barcode'] = bcs_b2
target_pts = (hipr_cell_info.loc[
                hipr_cell_info['cell_barcode'] == tax_code,
                ['centroid']
                ].values.squeeze())
target_pts = [list(c) for c in target_pts]
target_pts = np.rint(target_pts).astype(np.int64)

target_pts.shape
# %% md

# Get coordinates for spots

# %% codecell
ref_pts = [list(c) for c in mge_spot_seg_warp_props['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)

# %% md

# Get n nearest targets to reference points with distances

# %% codecell
r_max=10000  # maximum distance to measure density
nbrs = NearestNeighbors(radius=r_max, algorithm='kd_tree').fit(target_pts)
distances, indices = nbrs.radius_neighbors(ref_pts)

# %% md

# for each radial shell, calculate mean target density and stdev

# %% codecell
dr=1000  # Radial binning in pixels

r_s = np.arange(0, r_max, dr)
density_means = []
density_std = []
densities_all = []
for r in r_s:
    densities = []
    for dists in distances:
        count = sum((dists > r) * (dists < (r + dr)))
        area = np.pi * (r + dr)**2 - np.pi * r**2
        densities.append(count/area)
    densities_all.append(densities)
    density_means.append(np.mean(densities))
    density_std.append(np.std(densities))
densities_all = np.array(densities_all).T
densities_all.shape
# %% md

# Plot Radial density

# %% codecell
alpha=0.5
color=taxon_color_lookup.loc[
        taxon_color_lookup.sci_name == target_taxon,
        'tab20'
        ].values[0]
len(densities_all)
fig, ax = ip.general_plot()
for d in densities_all:
    ax.plot(r_s, d, color=color, alpha=0.1, lw=1)
ax.plot(r_s, density_means, color=color, lw=2, ls='--')
ax.set_ylabel('Taxon density')
ax.set_xlabel('Radial distance from spot')
ax.set_title(target_taxon)
# ax.fill_between(
#         r_s,
#         np.subtract(density_means, density_std),
#         np.add(density_means, density_std),
#         alpha=alpha,
#         color=color
#         )


# %% md

# Repeat for all taxa

# %% codecell
r_max=2500  # maximum distance to measure density
dr=250  # Radial binning in pixels
alpha=0.5

# get ordered list of taxa
codes_b10 = spot_props_roi['host_barcode'].unique()
# Get reference points
ref_pts = [list(c) for c in mge_spot_seg_warp_props['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
means_all = []
figa, axa = ip.general_plot()
for cb10 in codes_b10:
    # Get target points
    target_taxon=taxon_color_lookup.loc[
            taxon_color_lookup.code_b10 == cb10,
            'sci_name'
            ].values
    if target_taxon.shape[0] > 0:
        target_taxon = target_taxon[0]
        tax_code = taxon_color_lookup.loc[
                    taxon_color_lookup['sci_name'] == target_taxon,
                    'code'
                    ].values[0]
        target_pts = hipr_cell_info.loc[
                        hipr_cell_info['cell_barcode'] == tax_code,
                        ['centroid']
                        ].values.squeeze()
        target_pts = [list(c) for c in target_pts]
        target_pts = np.rint(target_pts).astype(np.int64)
        # Get nearest neighbors
        nbrs = NearestNeighbors(radius=r_max, algorithm='kd_tree').fit(target_pts)
        distances, indices = nbrs.radius_neighbors(ref_pts)
        # Get radial density values
        r_s = np.arange(0, r_max+dr, dr)
        density_means = []
        density_std = []
        densities_all = []
        for r in r_s:
            densities = []
            for dists in distances:
                count = sum((dists > r) * (dists < (r + dr)))
                area = np.pi * (r + dr)**2 - np.pi * r**2
                densities.append(count/area)
            densities_all.append(densities)
            density_means.append(np.mean(densities))
            density_std.append(np.std(densities))
        densities_all = np.array(densities_all).T
        # Plot
        color=taxon_color_lookup.loc[
                taxon_color_lookup.sci_name == target_taxon,
                'tab20'
                ].values[0]
        fig, ax = ip.general_plot()
        for d in densities_all:
            ax.plot(r_s, d, color=color, alpha=0.1, lw=1)
        ax.plot(r_s, density_means, color=color, lw=2, ls='--')
        ax.set_ylabel('Taxon density')
        ax.set_xlabel('Radial distance from spot')
        ax.set_title(target_taxon)
        output_basename = config['output_dir'] + '/spot_host_association/radial_density_histogram_genus_' + target_taxon
        ip.save_png_pdf(output_basename)
        axa.plot(r_s, density_means, color=color, label=target_taxon)

axa.legend()
axa.set_ylabel('Taxon density')
axa.set_xlabel('Radial distance from spot')
output_basename = config['output_dir'] + '/spot_host_association/radial_density_histogram_all'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()




# %% md

# Given a spot is located somewhere, how likely are you to find species x within radius R?

# %% codecell
# iterate through taxa
r_max=1000
dr=50
dims=(10,10)
ft=12
lw=1

fig, ax = ip.general_plot(dims=dims, ft=ft)
r_s = np.arange(0, r_max, dr)
# Get reference points
msswp_filt = mge_spot_seg_warp_props[
                mge_spot_seg_warp_props['max_intensity'] > filt_maxint
                ]
ref_pts = [list(c) for c in msswp_filt['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
probs_all = []
for cb10 in codes_b10:
    # Get target points
    target_taxon=taxon_color_lookup.loc[
            taxon_color_lookup.code_b10 == cb10,
            'sci_name'
            ].values
    if target_taxon.shape[0] > 0:
        # Get target taxon points
        target_taxon = target_taxon[0]
        tax_code = taxon_color_lookup.loc[
                    taxon_color_lookup['sci_name'] == target_taxon,
                    'code'
                    ].values[0]
        target_pts = hipr_cell_info.loc[
                        hipr_cell_info['cell_barcode'] == tax_code,
                        ['centroid']
                        ].values.squeeze()
        target_pts = [list(c) for c in target_pts]
        target_pts = np.rint(target_pts).astype(np.int64)
        # Get nearest neighbors
        nbrs = NearestNeighbors(radius=r_max, algorithm='kd_tree').fit(target_pts)
        distances, indices = nbrs.radius_neighbors(ref_pts)
        # Get radial probability
        prob_r = []
        for r in r_s:
            count = 0
            for dists in distances:
                count += any(dists < r)
            prob_r.append(count/ref_pts.shape[0])
        # Plot
        color=taxon_color_lookup.loc[
                taxon_color_lookup.sci_name == target_taxon,
                'tab20'
                ].values[0]
        ax.plot(r_s, prob_r, color=color, lw=lw, label=target_taxon)
        probs_all.append(prob_r)
ax.legend()
ax.set_ylabel('Probability of colocalization')
ax.set_xlabel('Radial distance from spot')
ax.set_title(target_taxon)
output_basename = config['output_dir'] + '/spot_host_association/radial_colocalization_probability'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()
plt.plot(r_s, np.sum(np.array(probs_all), axis=0))

# %% codecell
msswp_filt = mge_spot_seg_warp_props[
                mge_spot_seg_warp_props['max_intensity'] > filt_maxint
                ]
np.max(ref_pts)
np.max(ref_pts)
ref_pts = [list(c) for c in msswp_filt['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)

target_pts = hipr_cell_info.loc[
                :,
                ['centroid']
                ].values.squeeze()
target_pts = [list(c) for c in target_pts]
target_pts = np.rint(target_pts).astype(np.int64)
# nbrs = NearestNeighbors(algorithm='kd_tree').fit(target_pts)
# dist, ind = nbrs.kneighbors(ref_pts, n_neighbors=1)
# np.max(dist)

fig, ax = ip.general_plot()
ax.plot(target_pts[:,0],target_pts[:,1],'.')
ax.plot(ref_pts[:,0],ref_pts[:,1],'.')

# %% md

# =============================================================================
# ## Randomization of spot assigment in rois
# =============================================================================

# Extract cell id and barcode from each roi and combine

# %% codecell
hipr_seg_roi_props = pd.DataFrame([])

for i, (hipr_roi_fn, label) in tqdm(enumerate(zip(roi_npy_fns[sn_hipr], roi_npy_labels[sn_hipr]))):
    hipr_roi_pts_fn = re.sub('.npy','_withraw_resize_pts.csv', hipr_roi_fn)
    if os.path.exists(hipr_roi_pts_fn):
        # Get roi
        row = hipr_roi_props_df.iloc[i,:]
        bbox = eval(row['bbox'])
        hipr_roi = np.load(hipr_roi_fn)
        hipr_roi_bbox = hipr_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        # extract hiprfish seg roi
        hipr_seg_bbox = hipr_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        hipr_seg_roi = hipr_seg_bbox * 1*(hipr_roi_bbox > 0)
        # extract hipr barcode roi
        hipr_bc_bbox = hipr_bc[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        hipr_roi_bbox = hipr_roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        hipr_bc_roi = hipr_bc_bbox * (hipr_roi_bbox > 0)
        # get regoinprops on seg roi with barcode as raw image
        roi_props = sf.measure_regionprops(hipr_seg_roi, raw=hipr_bc_roi)
        roi_props['roi'] = label
        # Combine all regionprops from rois
        hipr_seg_roi_props = hipr_seg_roi_props.append(roi_props)
hsrp_fn = (config['output_dir'] +
                '/roi_masks/combined_outputs/'
                + sn_hipr + '_seg_roi_props.csv'
                )
hipr_seg_roi_props.to_csv(hsrp_fn)

# %% md

# iterative random assignment of spots to cells

# %% codecell
n = 10000

from random import sample
# Get total number of spots in rois
spot_intensities = spot_props_roi['max_intensity'].values.tolist()
# spot_intensities_full = np.zeros(hipr_seg_roi_props.shape[0])
zeros = [0]*(hipr_seg_roi_props.shape[0] - len(spot_intensities))
spot_intensities_full = spot_intensities + zeros
# spot_intensities_full[:spot_intensities.shape[0]] = spot_intensities
col_list = []
sp_list = []
for i in range(n):
    # Assign spots randomly to cells iteratively
    spot_intensities_full_rand = sample(
                                    spot_intensities_full,
                                    len(spot_intensities_full)
                                    )
    col_list.append('spot_intensity_' + str(i))
    sp_list.append(spot_intensities_full_rand)
sp_arr = np.array(sp_list).T
sp_df = pd.DataFrame(sp_arr, columns=col_list)
hipr_seg_roi_props = hipr_seg_roi_props.reset_index(drop=True)
hipr_seg_roi_props_randspotint = pd.concat([hipr_seg_roi_props, sp_df], axis=1)
hipr_seg_roi_props_randspotint.shape
# save random assignments
hsrpr_fn = (config['output_dir'] +
                '/roi_masks/combined_outputs/'
                + sn_hipr + '_seg_roi_props_randspotint.csv'
                )
hipr_seg_roi_props_randspotint.to_csv(hsrpr_fn)
hsrp_rand = hipr_seg_roi_props_randspotint

# %% md

# Plot number of spots vs taxon

# %% codecell
alpha=0.5

# for each taxon
for bc_tax in hipr_seg_roi_props['max_intensity'].unique():
    target_taxon=taxon_color_lookup.loc[
            taxon_color_lookup.code_b10 == bc_tax,
            'sci_name'
            ].values
    if target_taxon.shape[0] > 0:
        target_taxon = target_taxon[0]
        color=taxon_color_lookup.loc[
            taxon_color_lookup.sci_name == target_taxon,
            'tab20'
            ].values[0]
    else:
        target_taxon = 'Weird, not in probe design'
        color = (0,0,0)
    # Get indices of taxon in data frame
    ind_tax = np.where((hipr_seg_roi_props['max_intensity'] == bc_tax).values)[0]
    # Subset dataframe rows by taxon indices
    sp_arr_tax = sp_arr[ind_tax,:]
    # Count each column
    rand_counts = np.sum(sp_arr_tax > 0, axis=0)
    # plot distribution of spot assignment
    bins = np.arange(np.min(rand_counts), np.max(rand_counts))
    hist, bin_edges = np.histogram(rand_counts, bins=bins)
    x_vals = ip.get_line_histogram_x_vals(bin_edges)
    fig, ax = ip.general_plot()
    ax.plot(x_vals, hist, color=color)
    ax.fill_between(
            x_vals,
            np.zeros((x_vals.shape[0],)),
            hist,
            alpha=alpha,
            color=color
            )
    # PLot expected value
    ylims = ax.get_ylim()
    rand_count_mean = np.mean(rand_counts)
    ax.plot([rand_count_mean]*2, [0,0.5*ylims[1]], 'k')
    # plot location of actual assignment number
    true_count = spot_props_roi[
                    (spot_props_roi['host_barcode'] == bc_tax) &
                    (spot_props_roi['max_intensity'] > 0)
                    ].shape[0]
    ax.plot([true_count]*2, [0,0.5*ylims[1]], color=color)
    ax.set_xlabel('MGE association counts')
    ax.set_ylabel('Frequency')
    ax.set_title(target_taxon)
    output_basename = config['output_dir'] + '/spot_host_association/random_distribution_all_spot_count_' + target_taxon
    ip.save_png_pdf(output_basename)
    plt.show()
    plt.close()

# %% md

# plot bright spot intensity vs taxon

# %% codecell
alpha=0.5

# for each taxon
for bc_tax in hipr_seg_roi_props['max_intensity'].unique():
    target_taxon=taxon_color_lookup.loc[
            taxon_color_lookup.code_b10 == bc_tax,
            'sci_name'
            ].values
    if target_taxon.shape[0] > 0:
        target_taxon = target_taxon[0]
        color=taxon_color_lookup.loc[
            taxon_color_lookup.sci_name == target_taxon,
            'tab20'
            ].values[0]
    else:
        target_taxon = 'Weird, not in probe design'
        color = (0,0,0)
    # Get indices of taxon in data frame
    ind_tax = np.where((hipr_seg_roi_props['max_intensity'] == bc_tax).values)[0]
    # Subset dataframe rows by taxon indices
    sp_arr_tax = sp_arr[ind_tax,:]
    # Count each column
    rand_counts = np.sum(sp_arr_tax > filt_maxint, axis=0)
    # plot distribution of spot assignment
    bins = np.arange(np.min(rand_counts), np.max(rand_counts))
    hist, bin_edges = np.histogram(rand_counts, bins=bins)
    x_vals = ip.get_line_histogram_x_vals(bin_edges)
    fig, ax = ip.general_plot()
    ax.plot(x_vals, hist, color=color)
    ax.fill_between(
            x_vals,
            np.zeros((x_vals.shape[0],)),
            hist,
            alpha=alpha,
            color=color
            )
    # PLot expected value
    ylims = ax.get_ylim()
    rand_count_mean = np.mean(rand_counts)
    ax.plot([rand_count_mean]*2, [0,0.5*ylims[1]], 'k')
    # plot location of actual assignment number
    true_count = spot_props_roi[
                    (spot_props_roi['host_barcode'] == bc_tax) &
                    (spot_props_roi['max_intensity'] > filt_maxint)
                    ].shape[0]
    ax.plot([true_count]*2, [0,0.5*ylims[1]], color=color)
    ax.set_xlabel('Bright spot association counts')
    ax.set_ylabel('Frequency')
    ax.set_title(target_taxon)
    output_basename = config['output_dir'] + '/spot_host_association/random_distribution_bright_spot_count_' + target_taxon
    ip.save_png_pdf(output_basename)
    plt.show()
    plt.close()

# %% md

# plot dim spot intensity vs taxon

# %% codecell
bins=100
alpha=0.5

for bc_tax in hipr_seg_roi_props['max_intensity'].unique():
    target_taxon=taxon_color_lookup.loc[
            taxon_color_lookup.code_b10 == bc_tax,
            'sci_name'
            ].values
    if target_taxon.shape[0] > 0:
        target_taxon = target_taxon[0]
        color=taxon_color_lookup.loc[
            taxon_color_lookup.sci_name == target_taxon,
            'tab20'
            ].values[0]
    else:
        target_taxon = 'Weird, not in probe design'
        color = (0,0,0)
    # Get indices of taxon in data frame
    ind_tax = np.where((hipr_seg_roi_props['max_intensity'] == bc_tax).values)[0]
    # Subset dataframe rows by taxon indices
    sp_arr_tax = sp_arr[ind_tax,:]
    # Count each column
    rand_counts = np.sum((sp_arr_tax <= filt_maxint) & (sp_arr_tax > 0), axis=0)
    # plot distribution of spot assignment
    bins = np.arange(np.min(rand_counts), np.max(rand_counts))
    hist, bin_edges = np.histogram(rand_counts, bins=bins)
    x_vals = ip.get_line_histogram_x_vals(bin_edges)
    fig, ax = ip.general_plot()
    ax.plot(x_vals, hist, color=color)
    ax.fill_between(
            x_vals,
            np.zeros((x_vals.shape[0],)),
            hist,
            alpha=alpha,
            color=color
            )
    # PLot expected value
    ylims = ax.get_ylim()
    rand_count_mean = np.mean(rand_counts)
    ax.plot([rand_count_mean]*2, [0,0.5*ylims[1]], 'k')
    # plot location of actual assignment number
    true_count = spot_props_roi[
                    (spot_props_roi['host_barcode'] == bc_tax) &
                    (spot_props_roi['max_intensity'] <= filt_maxint) &
                    (spot_props_roi['max_intensity'] > 0)
                    ].shape[0]
    ax.plot([true_count]*2, [0,0.5*ylims[1]], color=color)
    ax.set_xlabel('Dim spot association counts')
    ax.set_ylabel('Frequency')
    ax.set_title(target_taxon)
    output_basename = config['output_dir'] + '/spot_host_association/random_distribution_dim_spot_count_' + target_taxon
    ip.save_png_pdf(output_basename)
    plt.show()
    plt.close()

# %% md

# =============================================================================
# ## Single cell PCA
# =============================================================================

# Get the nearest neighbors dict for each cell with distances

# %% codecell
hipr_cell_info = hipr_cell_info.reset_index()
# %% codecell
r_um=5  # um range

r_pix = r_um / res_mge
# Get points
cell_coords = hipr_cell_info['centroid'].values
cell_coords = [list(c) for c in cell_coords]
cell_coords = np.rint(cell_coords).astype(np.int64)
# Get nearest neighbors
nbrs = NearestNeighbors(radius=r_pix, algorithm='kd_tree').fit(cell_coords)
distances, indices = nbrs.radius_neighbors(cell_coords)

# %% md

# Get the taxon assignemnt dict for each cell

# %% codecell
# Convert cell row index to taxid
ind_tax_dict = dict(zip(hipr_cell_info.index, hipr_cell_info['cell_barcode_b10']))
# convert taxon id to column index in matrix
tax_list = hipr_cell_info['cell_barcode_b10'].unique().tolist()
tax_ind_dict = dict(zip(tax_list, np.arange(len(tax_list))))

# for cid, nn_arr in cid_nn_dict.items():
#     taxs = []
#     for ind, _ in nn_arr:
#         taxs.append(ind_tax_dict[ind])
#     cid_nn_dict[cid] = np.hstack([nn_arr,taxs])

# %% md

# Construct the cell x feature matrix

# %% codecell
cell_feature_matrix = np.zeros((len(distances), len(tax_list)))
for i, (dists, inds) in enumerate(zip(distances, indices)):  # Get row
    for dist, ind in zip(dists, inds):
        if dist > 0:  # Skip self
            tax = ind_tax_dict[ind]
            j = tax_ind_dict[tax]  # Get column
            cell_feature_matrix[i,j] += 1/dist  # weight count by distance


# # Standardize cell id row index in matrix
# cid_list = hipr_cell_info['label'].values.tolist()
# # Standardize taxon column index in matrix
# tax_list = hipr_cell_info['cell_barcode_b10'].unique().tolist()
# tax_ind_dict = dict(zip(taxon_list, np.arange(len(tax_list))))
# # initialize matrix
# cell_feature_matrix = np.zeros(len(cid_nn_dict), len(tax_list))
# for i, cid in tqdm(enumerate(cid_list)):  # Get row
#     nn_arr = cid_nn_dict[cid]
#     for index, dist, tax in nn_arr:
#         j = tax_ind_dict[tax]  # Get column
#         cell_feature_matrix[i,j] += 1/dist  # weight count by distance


# %% md

# Run PCA

# %% codecell
from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='randomized', random_state=42)
pca.fit(cell_feature_matrix)
cell_feature_pca = pca.transform(cell_feature_matrix)
pca.explained_variance_ratio_
# %% codecell
fig, ax = ip.general_plot()
ax.bar(np.arange(5),pca.explained_variance_ratio_)
ax.set_xticklabels(['0','1','2','3','4','5'])
ax.set_xlabel('PC')
ax.set_ylabel('Percent of variance')
output_basename = config['output_dir'] + '/spot_host_association/pca_r_10_distweighted_pov_bar'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()


# %% codecell
# PLot PC 1 and 2
ms=1
dims=(5,5)
c='k'
alpha=0.1

fig, ax = ip.general_plot(dims=dims)
ax.scatter(cell_feature_pca[:,0], cell_feature_pca[:,1], s=ms, c=c, alpha=alpha)
# ax.set_xlim(-0.65,1.15)
# ax.set_ylim(-0.9,0.9)
# ax.set_xticks([-0.5,0,0.5,1])
# ax.set_yticks([-0.5,0,0.5])
output_basename = config['output_dir'] + '/spot_host_association/pca_r_10_distweighted_hipr_unlabeled'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()


# %% codecell
# PC 1 and 2 info
comp = pca.components_[:2]
inds = [np.argpartition(c, -4)[-4:] for c in comp]
vals = [c[i] for c, i in zip(comp, inds)]
inds
vals
# %% codecell
cm_ind_tax_dict = dict(zip(np.arange(len(tax_list)), tax_list))
tax_sciname_dict = dict(zip(taxon_color_lookup['code_b10'], taxon_color_lookup['sci_name']))

names = [[tax_sciname_dict[cm_ind_tax_dict[i]] for i in i_] for i_ in inds]
names

# %% md

# Construct the cell x feature matrix but without distance weighting

# %% codecell
cell_feature_matrix = np.zeros((len(distances), len(tax_list)))
for i, (dists, inds) in enumerate(zip(distances, indices)):  # Get row
    for dist, ind in zip(dists, inds):
        if dist > 0:  # Skip self
            tax = ind_tax_dict[ind]
            j = tax_ind_dict[tax]  # Get column
            cell_feature_matrix[i,j] += 1  # weight count by distance

pca = PCA(n_components=5, svd_solver='randomized', random_state=42)
pca.fit(cell_feature_matrix)
cell_feature_pca = pca.transform(cell_feature_matrix)
print('pca.explained_variance_ratio_',pca.explained_variance_ratio_)

fig, ax = ip.general_plot()
ax.bar(np.arange(5),pca.explained_variance_ratio_)
ax.set_xticklabels(['0','1','2','3','4','5'])
ax.set_xlabel('PC')
ax.set_ylabel('Percent of variance')
output_basename = config['output_dir'] + '/spot_host_association/pca_r_10_unweighted_pov_bar'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()

ms=1
dims=(5,5)
c='k'
alpha=0.1

fig, ax = ip.general_plot(dims=dims)
ax.scatter(cell_feature_pca[:,0], cell_feature_pca[:,1], s=ms, c=c, alpha=alpha)
# ax.set_xlim(-0.65,1.15)
# ax.set_ylim(-0.9,0.9)
# ax.set_xticks([-0.5,0,0.5,1])
# ax.set_yticks([-0.5,0,0.5])
output_basename = config['output_dir'] + '/spot_host_association/pca_r_10_unweighted_hipr_unlabeled'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()

comp = pca.components_[:2]
inds = [np.argpartition(c, -4)[-4:] for c in comp]
vals = [c[i] for c, i in zip(comp, inds)]
cm_ind_tax_dict = dict(zip(np.arange(len(tax_list)), tax_list))
tax_sciname_dict = dict(zip(taxon_color_lookup['code_b10'], taxon_color_lookup['sci_name']))
names = [[tax_sciname_dict[cm_ind_tax_dict[i]] for i in i_] for i_ in inds]
print('names',names)
print('vals',vals)

# %% md

# Color by taxon

# %% codecell
ms=1
dims=(5,5)
alpha=0.5

tax_col_dict = dict(zip(taxon_color_lookup['code_b10'], taxon_color_lookup['tab20']))
col_list = [tax_col_dict[t] if t != 35 else (0,0,1) for t in hipr_cell_info['cell_barcode_b10'].values]
fig, ax = ip.general_plot(dims=dims)
ax.scatter(cell_feature_pca[:,0], cell_feature_pca[:,1], s=ms, c=col_list, alpha=alpha)
# ax.set_xlim(-0.65,1.15)
# ax.set_ylim(-0.9,0.9)
# ax.set_xticks([-0.5,0,0.5,1])
# ax.set_yticks([-0.5,0,0.5])
output_basename = config['output_dir'] + '/spot_host_association/pca_r_10_unweighted_hipr_taxlabeled'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()


# %% md

# ### Repeat for spots

# %% codecell
mge_spot_seg_warp_props = mge_spot_seg_warp_props.reset_index()

# %% codecell
spot_coords = [list(c) for c in mge_spot_seg_warp_props['centroid'].values]
spot_coords = np.rint(spot_coords).astype(np.int64)
distances, indices = nbrs.radius_neighbors(spot_coords)

spot_feature_matrix = np.zeros((len(distances), len(tax_list)))
for i, (dists, inds) in enumerate(zip(distances, indices)):  # Get row
    for dist, ind in zip(dists, inds):
        tax = ind_tax_dict[ind]
        j = tax_ind_dict[tax]  # Get column
        spot_feature_matrix[i,j] += 1

spot_feature_pca = pca.transform(spot_feature_matrix)

dims=(5,5)
ms_sp=2
# c_sp='m'
# alpha_sp=1
cmap='cividis_r'

ms_cl=1
c_cl=plt.cm.get_cmap(cmap)(0)
alpha_cl=0.25

# Get a colormap for the spots based on the spot intensity
maxints = mge_spot_seg_warp_props['max_intensity']
normvals = maxints.values / maxints.max()
cmap=plt.cm.get_cmap(cmap)
c_sp = [cmap(v) for v in normvals]
# also adjust the plotted alpha based on spot intensity
adj=0.5  # value of 0.5 means 50% of the spots have alpha 1 and the rest spread between
adj = 1/adj
alpha_sp = normvals*adj * (normvals*adj < 1) + (normvals*adj > 1)*1
# Now plot
fig, ax = ip.general_plot(dims=dims)
ax.scatter(cell_feature_pca[:,0], cell_feature_pca[:,1], s=ms_cl, color=c_cl, alpha=alpha_cl)
ax.scatter(spot_feature_pca[:,0], spot_feature_pca[:,1], s=ms_sp, c=c_sp, alpha=alpha_sp)
# ax.set_xlim(-0.65,1.15)
# ax.set_ylim(-0.9,0.9)
# ax.set_xticks([-0.5,0,0.5,1])
# ax.set_yticks([-0.5,0,0.5])
output_basename = config['output_dir'] + '/spot_host_association/pca_r_10_unweighted_hipr_unlabeled_mge'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()


# %% md

# ### Clustering

# Run clustering

# %% codecell
n_clust = 20

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clust, random_state=42).fit(cell_feature_pca)

# %% md

# show clustering on PCA projection

# %% codecell
ms=1
dims=(5,5)
alpha=0.5
cmap='tab20'

# Get colormap for the clusters
labels = kmeans.labels_
cols = plt.cm.get_cmap(cmap).colors[:len(labels)]
lab_col_dict = dict(zip(np.unique(labels), cols))
col_list = [lab_col_dict[l] for l in labels]

ms_sp=2
cmap='binary'
alpha_sp=1

# Get a colormap for the spots based on the spot intensity
maxints = mge_spot_seg_warp_props['max_intensity']
normvals = maxints.values / maxints.max()
adj=0.75  # value of 0.25 means 75% of the spots have alpha 1 and the rest spread between
adj = 1/adj
normvals = normvals*adj * (normvals*adj < 1) + (normvals*adj > 1)*1
cmap=plt.cm.get_cmap(cmap)
c_sp = [cmap(v) for v in normvals]

fig, ax = ip.general_plot(dims=dims)
ax.scatter(cell_feature_pca[:,0], cell_feature_pca[:,1], s=ms, c=col_list, alpha=alpha)
ax.scatter(spot_feature_pca[:,0], spot_feature_pca[:,1], s=ms_sp, c=c_sp, alpha=alpha_sp)
# ax.set_xlim(-0.65,1.15)
# ax.set_ylim(-0.9,0.9)
# ax.set_xticks([-0.5,0,0.5,1])
# ax.set_yticks([-0.5,0,0.5])
output_basename = config['output_dir'] + '/spot_host_association/pca_r_10_unweighted_hipr_clustlabeled'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()

# %% md

# Assign spots to clusters

# %% codecell
labels_sp = kmeans.predict(spot_feature_pca)
col_list_sp = [lab_col_dict[l] for l in labels_sp]

maxints = mge_spot_seg_warp_props['max_intensity']
normvals = maxints.values / maxints.max()
adj=1  # value of 0.25 means 75% of the spots have alpha of 1
adj = 1/adj
alpha_sp = normvals*adj * (normvals*adj < 1) + (normvals*adj > 1)*1

fig, ax = ip.general_plot(dims=dims)
ax.scatter(spot_feature_pca[:,0], spot_feature_pca[:,1], s=ms_sp, c=col_list_sp, alpha=alpha_sp)

# %% md

# Show heatmap

# %% codecell
labels[...,np.newaxis].shape
cell_feature_matrix_clust = np.hstack([labels[...,np.newaxis], cell_feature_matrix])
cell_feature_matrix_clust_sort = np.empty((1,cell_feature_matrix_clust.shape[1]))
for i in range(n_clust):
    sub = cell_feature_matrix_clust[cell_feature_matrix_clust[:,0] == i]
    cell_feature_matrix_clust_sort = np.vstack([cell_feature_matrix_clust_sort, sub])
plt.imshow(cell_feature_matrix_clust_sort.T, cmap='viridis')
plt.axis('tight')
plt.colorbar()

# %% md

# Show mean region for different clusters of spots

# %% codecell
sfm_clust = np.hstack([labels_sp[...,np.newaxis], spot_feature_matrix])
sfm_clust_means = np.empty((1,sfm_clust.shape[1]))
for i in range(n_clust):
    sub = np.mean(sfm_clust[sfm_clust[:,0] == i] + 1e-15, axis=0)
    sfm_clust_means = np.vstack([sfm_clust_means, sub])




sci_lab = [tax_sciname_dict[l] if l != 35 else 'weird' for l in tax_list]
sfmcm_df = pd.DataFrame(sfm_clust_means, columns = ['clust'] + sci_lab)
bool = (sfmcm_df.sum(axis=0) > 5)
sfmcm_df_sub = sfmcm_df.loc[:, bool[bool].index.values]
plt.imshow(sfmcm_df_sub.values.T)
plt.axis('tight')
plt.colorbar()
plt.yticks(ticks=np.arange(sfmcm_df_sub.shape[1]),labels=sfmcm_df_sub.columns)

# %% md

# Show clusters on spatial map

# %% codecell
# Combine cluster labels with cell id labels
# re-color cell ids with cluster colors



# %% codecell


















# %% md

# =============================================================================
# ## Old
# =============================================================================



# %% codecell
# Show on mge raw
# Load mge fish tiff
im_inches=10
outline_col = (1,1,0)
ft = 20

mge_tiff_fn = config['output_dir'] + '/' + sn_mge + '_c1-2.tif'
mge_tiff = np.array(Image.open(mge_tiff_fn))
mge_tiff.shape
fig, ax, cbar = ip.plot_image(mge_tiff, im_inches=im_inches)
mge_roi_props_list = []
# Get region properties
for i, fn in tqdm(enumerate(roi_npy_fns[sn_mge])):
    roi = np.load(fn)
    props_ = sf.measure_regionprops(roi, raw=np.zeros(roi.shape))
    mge_roi_props_list.append(props_.values.tolist())
    # Show segmentations on Raw
    ax = ip.plot_seg_outline(ax, roi, col=outline_col)
    # Add label number at centroid
    ax.text(props_['centroid-1'].values[0], props_['centroid-0'].values[0],
                s=str(i+1), size=ft)
# Save image
output_basename = config['output_dir'] + '/' + sn_mge + '_rois'
ip.save_png_pdf(output_basename)
