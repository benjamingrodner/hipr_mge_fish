# %% md

# # Figure 5c: Phage TermL2 stain HiPR MGE FISH

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
import umap
from cv2 import resize, INTER_CUBIC, INTER_NEAREST
from sklearn.neighbors import NearestNeighbors
from random import sample
from collections import defaultdict
from copy import copy

gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5c/MGEFISH'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %%

# Go into your configuration file and adjust all of the 'Paths' so they are relative to the workdir.

# Also adjust the 'Inputs' parameters to fit the images.

# %% codecell
config_fn = 'config_mgefish.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

config_hipr_fn = '../config_hiprfish.yaml' # relative path to config file from workdir

with open(config_hipr_fn, 'r') as f:
    config_hipr = yaml.safe_load(f)
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
import fn_manual_rois as fmr
from fn_face_morpher import warp_image, weighted_average_points

# %% md

# =============================================================================
# ## Preparing images for FIJI
# =============================================================================

# ### Prepare HiPRFISH image for manual annotation

# Load hiprfish registered image and get the max projection

# %% codecell
hipr_sn = '2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_3_mode_spec'
hipr_raw_fn = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_registered.npy'
hipr_raw = np.load(hipr_raw_fn)
hipr_raw_max = np.max(hipr_raw, axis=2)

del hipr_raw

# %% md

# Rescale the max projection so that each pixel has the same resolution as MGE image

# Then save as a tif image

# %% codecell
# Resize
mge_res = config['resolution']
hipr_res = config_hipr['resolution']
factor_resize = hipr_res / mge_res
hipr_raw_max_resize = resize(
        hipr_raw_max,
        None,
        fx = factor_resize,
        fy = factor_resize,
        interpolation = INTER_NEAREST
        )

# %% codecell
# to tiff and npy
hipr_raw_max_resize_fn = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_max_resize.npy'
np.save(hipr_raw_max_resize_fn, hipr_raw_max_resize)
hipr_pil = Image.fromarray(hipr_raw_max_resize)
hipr_pil_fn = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_max_resize.tif'
hipr_pil.save(hipr_pil_fn)

# %% md

# =============================================================================
# ## Register full images by translation only
# =============================================================================

# For the MGEFISH image, use the Zen software to export a single channel tif

# ### Point to point registration

# Get global registration points for sister features in the exact same order on each image using FIJI

# ...

# Now Load the points

# %% codecell
hipr_regpts_fn = config['output_dir'] + '/../sister_regions/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_3_mode_spec_max_resize_registrationpoints.csv'
mge_regpts_fn = config['output_dir'] + '/../sister_regions/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_1_mode_airy_Airyscan_Processing_shad_stitch_c3_registrationpoints.csv'
hipr_regpts = pd.read_csv(hipr_regpts_fn)
mge_regpts = pd.read_csv(mge_regpts_fn)

# %% md

# Use the snakemake rule to convert the MGEFISH czi file to npy

# %% codecell
# mge_sn = '2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_1_mode_airy_Airyscan_Processing_shad_stitch'
# mge_raw_fn =
input_filenames = glob.glob(config['input_dir'] + '/' + config['input_regex']
                            + config['input_ext'])
input_filenames.sort()
print(len(input_filenames))
input_filenames
# %% codecell
input_fns_split = [os.path.split(fn)[1] for fn in input_filenames]
sample_names = [re.sub(config['input_ext'], '', fn) for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['workdir'] + '/' + config['input_table_fn'], index=False)
input_table.values

# %% codecell
dry_run = False  # Just create DAG if True
n_cores = 1  # number of allowed cores for the snakemake to use
force_run = False  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['snakefile_convert']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
command = " ".join(['snakemake', '-s', snakefile, '--configfile', config_fn, '-j',
                    str(n_cores), dr, fr])

run_fn = config['workdir'] + '/run_{}.sh'.format(snakefile)
with open(run_fn, 'w') as f:
    f.write(command)

command

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5c/MGEFISH
# $ sh run_Snakefile_convert.sh
# ```

# Now load the MGEFISH image and get the cell channel

# %% codecell
mge_sn = input_table['sample_name'].values[0]
mge_raw_fmt = config['output_dir'] + '/' + config['raw_fmt']
mge_raw = ip.load_output_file(config, 'raw_fmt', mge_sn)
mge_cell = mge_raw[:,:,config['cell_seg']['channels'][0]]

# %% md

# Get the shifts between points

# %% codecell
n=10000
min_diff=0.01
step_reduce=0.75

hipr_pts = hipr_regpts.loc[:,['Y','X']].values
mge_pts = mge_regpts.loc[:,['Y','X']].values
# mge_pts_pix = mge_pts / mge_res
shifts = fmr.register_points_mindist(
        hipr_pts,mge_pts,
        n=n,
        min_diff=min_diff,
        step_reduce=step_reduce
        )
shifts

# %% md

# Show the shifted images

# %% codecell
# Get the max dimensions on both images
mge_shp = mge_cell.shape
hipr_shp = hipr_raw_max_resize.shape
maxdims = np.max(np.array([list(mge_shp), list(hipr_shp)]), axis=0)
# Add the shift values to the max dimensions
merge_shp = maxdims + np.abs(shifts)
# Place the shifted image
mge_cell_shift = np.zeros(merge_shp)
ulc = np.where(shifts<0, 0, shifts)  # place upper left corner at 0 if shift is negative
mge_cell_shift[ulc[0]:mge_shp[0]+ulc[0],ulc[1]:mge_shp[1]+ulc[1]] = mge_cell
# place the unshiffted image
hipr_max_shift = np.zeros(merge_shp)
nshifts = -1*shifts
ulc = np.where(nshifts<0, 0, nshifts)  # place upper left corner at 0 if shift is negative
hipr_max_shift[ulc[0]:hipr_shp[0]+ulc[0],ulc[1]:hipr_shp[1]+ulc[0]] = hipr_raw_max_resize

# %% codecell
# Show the overlay
mcs_norm = ip.zero_one_thresholding(mge_cell_shift, clims=(0.01,0.225))
plt.imshow(mcs_norm, cmap='inferno')
plt.show()
plt.close()
hms_norm = ip.zero_one_thresholding(hipr_max_shift, clims=(0.01,0.3))
plt.imshow(hms_norm, cmap='inferno')

# %% codecell
overlay_shift = np.dstack([mcs_norm, hms_norm, np.zeros(hms_norm.shape)])
image_list = [mcs_norm, hms_norm, overlay_shift]
cmaps = ['inferno','inferno','']
ip.subplot_square_images(image_list,(1,3),cmaps=cmaps)
output_basename = os.path.split(hipr_regpts_fn)[0] + '/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_hiprmge_overlay_cell'
ip.save_png_pdf(output_basename)

# %% md

# =============================================================================
# ## Manual annotation of sister regions
# =============================================================================

# ### Manually annotate sister regions between the two images and save a mask for each region

# ...

# Now convert tiff mask files to npy and get a list of roi filenames

# %% codecell
# write roi tiffs to npy files
Image.MAX_IMAGE_PIXELS = None
roi_dir = config['output_dir'] + '/../sister_regions'
sample_names = [mge_sn, hipr_sn]
roi_npy_fns = {}
roi_npy_labels = {}
# process for mge and hipr images
for sn in sample_names:
    roi_sn_dir = roi_dir
    # Load tiffs
    roi_tiff_fns = glob.glob(roi_sn_dir + '/*.tif')
    roi_tiff_fns.sort()
    roi_npy_fns[sn] = []
    roi_npy_labels[sn] = []
    for fn in tqdm(roi_tiff_fns):
        if sn in fn:
            im = np.array(Image.open(fn))
            # save as numpy files
            out_fn = re.sub('.tif','.npy',fn)
            np.save(out_fn, im)
            roi_npy_fns[sn] += [out_fn]
            label = re.search(r'(?<=sister_)\d+', fn).group(0)
            roi_npy_labels[sn] += [label]

# %% md

# Prepare sister regions for manual selection of registration feature points

# %% codecell
# Get MGE roi regionprops
mge_roi_props_list = []
# Get region properties
for sn in sample_names:
    props_list = []
    for i, fn in tqdm(enumerate(roi_npy_fns[sn])):
        roi = 1*(np.load(fn) > 0)
        props_ = sf.measure_regionprops(roi, raw=np.zeros(roi.shape))
        props_list.append(props_)
        bbox = props_['bbox'].values[0]
        roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]

    props_df = pd.concat(props_list)
    props_df_fn = roi_dir + '/' + sn + '_sister_props.csv'
    props_df.to_csv(props_df_fn)

# %% codecell
# Save raw images masked for each sister region
raw_images = [mge_cell, hipr_raw_max_resize]
for sn, im in zip(sample_names, raw_images):
    props_df_fn = roi_dir + '/' + sn + '_sister_props.csv'
    props_df = pd.read_csv(props_df_fn)
    for i in tqdm(range(props_df.shape[0])):
        row = props_df.iloc[i,:]
        bbox = eval(row['bbox'])
        roi_fn = roi_npy_fns[sn][i]
        roi = np.load(roi_fn)
        print(roi.shape)
        im_bbox = im[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        print(im.shape)
        roi_bbox = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        im_bbox_roi = im_bbox * 1*(roi_bbox > 0)
        output_fn = re.sub('.npy','_withraw.npy', roi_fn)
        np.save(output_fn, im_bbox_roi)
        output_fn = re.sub('.npy','_withraw.tiff', roi_fn)
        im_pil = Image.fromarray(im_bbox_roi)
        im_pil.save(output_fn)
        # print('Wrote:', output_fn)

# %% md

# ### Correlation between spots and taxa in sister regions

# Get roi spot counts

# removed unused rois so that mge and hipr are the same length

# Need to reproduce and reorder roi basenames so they match sisters

# %% codecell
roi_labels = {}
roi_basenames = {}
for sn in sample_names:
    roi_sn_dir = roi_dir
    # Load tiffs
    roi_tiff_fns = glob.glob(roi_dir + '/*.tif')
    roi_tiff_fns.sort()
    roi_basenames[sn] = []
    roi_labels[sn] = []
    for fn in tqdm(roi_tiff_fns):
        if sn in fn:
            # im = np.array(Image.open(fn))
            # save as numpy files
            out_basename = re.sub('.tif','',fn)
            # np.save(out_fn, im)
            roi_basenames[sn] += [out_basename]
            label = re.search(r'(?<=sister_)\d+', fn).group(0)
            roi_labels[sn] += [label]

print(roi_labels)
print(roi_basenames)
print([len(roi_basenames[sn]) for sn in sample_names])

all([i == j for i, j in zip(roi_labels[mge_sn], roi_labels[hipr_sn])])

# %% md

# Set an intensity threshold for defining bright spots and dim spots

# Pick the 633 spot channel

# %% codecell
# Pick channel, load spot properties
s_ch = 3
mge_spot_props = ip.load_output_file(
        config, 'spot_props_max_split_fmt',
        sample_name=mge_sn, spot_chan=s_ch
        )

# %% codecell
# Select threshold
thr=0.0045

spot_int = mge_spot_props.max_intensity.values
spot_int = np.sort(spot_int)
fig, ax = ip.general_plot()
ax.bar(np.arange(spot_int.shape[0]), spot_int)
ax.set_ylim(0,0.02)
ax.plot([0,spot_int.shape[0]], [thr]*2)

# %% md

# Get spot seg properties for each roi

# %% codecell
# Reproduce npy files after removign extra rois
for sn in sample_names:
    for i, (bn, lab) in tqdm(enumerate(zip(roi_basenames[sn], roi_labels[sn]))):
        fn = bn + '.tif'
        roi_pil = Image.open(fn)
        roi_tif = np.array(roi_pil)
        npy_fn = bn + '.npy'
        np.save(npy_fn, roi_tif)

# %% codecell
# Reproduce roi regionprops after reomoving extra rois
for sn in sample_names:
    props_list = []
    for i, (bn, lab) in tqdm(enumerate(zip(roi_basenames[sn], roi_labels[sn]))):
        fn = bn + '.npy'
        roi = 1*(np.load(fn) > 0)
        props_ = sf.measure_regionprops(roi, raw=np.zeros(roi.shape))
        props_['roi'] = lab
        props_list.append(props_)
        bbox = props_['bbox'].values[0]
        roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]

    props_df = pd.concat(props_list)
    props_df_fn = roi_dir + '/' + sn + '_sister_props_filtrois.csv'
    props_df.to_csv(props_df_fn)


# %% codecell
# Get mge roi props
mge_spot_seg = ip.load_output_file(
        config, 'spot_seg_max_split_fmt',
        sample_name=mge_sn, spot_chan=s_ch
        )

mge_spot = mge_raw[:,:,s_ch]

mge_roi_basenames = roi_basenames[mge_sn]
mge_roi_props_fn = roi_dir + '/' + mge_sn + '_sister_props_filtrois.csv'
mge_roi_props = pd.read_csv(mge_roi_props_fn)
for i, bn in tqdm(enumerate(mge_roi_basenames)):
    # Load roi and props
    roi_fn = bn + '.npy'
    roi = np.load(roi_fn)
    # There is a bug in javabridge where it doesn't load the stitched file shape, but rather the unstitched larger shape.
    roi_extend = np.zeros(mge_cell.shape)
    rsh = roi.shape
    roi_extend[:rsh[0],:rsh[1]] = roi
    # Get bounding box
    row = mge_roi_props.iloc[i,:]
    bbox = eval(row['bbox'])
    roi_box = roi_extend[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    raw_box = mge_spot[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    seg_box = mge_spot_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    # Masking
    raw_box_roi = raw_box * 1*(roi_box > 0)
    seg_box_roi = seg_box * 1*(roi_box > 0)
    # Properties
    seg_box_roi_props = sf.measure_regionprops(seg_box_roi, raw=raw_box_roi)
    sbrp_fn = bn + '_props.csv'
    seg_box_roi_props.to_csv(sbrp_fn, index=False)


# %% md

# Get roi properties for hiprfish ident

# %% codecell
# Resize barcode and segmentation image
hipr_bc_fn = '../' + config_hipr['output_dir'] + '/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_3_mode_spec_identification_barcode.npy'
hipr_seg_fn = '../' + config_hipr['output_dir'] + '/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_3_mode_spec_seg.npy'
hipr_bc = np.load(hipr_bc_fn)
hipr_seg = np.load(hipr_seg_fn)
hipr_bc_resize = resize(
                    hipr_bc, None, fx = factor_resize,
                    fy = factor_resize, interpolation = INTER_NEAREST
                    )
hipr_seg_resize = resize(
                    hipr_seg, None, fx = factor_resize,
                    fy = factor_resize, interpolation = INTER_NEAREST
                    )
hipr_bc_resize_fn = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_barcode_resized.npy'
hipr_seg_resize_fn = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_seg_resized.npy'
hipr_sbrp_fn = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_seg_bc_props_resized.csv'
np.save(hipr_bc_resize_fn, hipr_bc_resize)
np.save(hipr_seg_resize_fn, hipr_seg_resize)

# %% codecell
# Get hipr roi properties
hipr_roi_basenames = roi_basenames[hipr_sn]
hipr_roi_props_fn = roi_dir + '/' + hipr_sn + '_sister_props_filtrois.csv'
hipr_roi_props = pd.read_csv(hipr_roi_props_fn)
for i, bn in tqdm(enumerate(hipr_roi_basenames)):
    # Load roi and props
    roi_fn = bn + '.npy'
    roi = np.load(roi_fn)
    # There is a bug in javabridge where it doesn't load the stitched file shape, but rather the unstitched larger shape.
    roi_extend = np.zeros(hipr_seg_resize.shape)
    rsh = roi.shape
    roi_extend[:rsh[0],:rsh[1]] = roi
    # Get bounding box
    row = hipr_roi_props.iloc[i,:]
    bbox = eval(row['bbox'])
    roi_box = roi_extend[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    raw_box = hipr_bc_resize[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    seg_box = hipr_seg_resize[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    # Masking
    raw_box_roi = raw_box * 1*(roi_box > 0)
    seg_box_roi = seg_box * 1*(roi_box > 0)
    # Properties
    seg_box_roi_props = sf.measure_regionprops(seg_box_roi, raw=raw_box_roi)
    sbrp_fn = bn + '_props.csv'
    seg_box_roi_props.to_csv(sbrp_fn, index=False)

# %% codecell
i=56
roi_fn = hipr_roi_basenames[i] + '.npy'
roi = np.load(roi_fn)
roi.shape
row = hipr_roi_props.iloc[i,:]
bbox = eval(row['bbox'])
roi_box = roi[bbox[0]:bbox[2],bbox[1]:bbox[3]]
raw_box = hipr_bc_resize[bbox[0]:bbox[2],bbox[1]:bbox[3]]

plt.imshow(raw_box * 1*(roi_box > 0))
plt.imshow(hipr_bc_resize * 1*(roi > 0))

roi_tiff_fn = hipr_roi_basenames[i] + '.tif'
roi_pil = Image.open(roi_tiff_fn)
roi_tif = np.array(roi_pil)
roi_tif.shape
plt.imshow(hipr_bc_resize * 1*(roi_tif > 0))
np.save(roi_fn, roi_tif)

# %% md

# Count spots and taxa

# %% codecell
# mge_roi_labels = roi_labels[mge_sn]
roi_count_dict = defaultdict(list) # For recording data
taxa_list = np.unique(hipr_bc_resize)
for i, (mge_bn, hipr_bn) in tqdm(enumerate(zip(mge_roi_basenames, hipr_roi_basenames))):
    # Load props
    mp_fn = mge_bn + '_props.csv'
    mge_props = pd.read_csv(mp_fn)
    hp_fn = hipr_bn + '_props.csv'
    hipr_props = pd.read_csv(hp_fn)
    # Count spots normalized by number of cells
    n_cells = hipr_props.shape[0]
    if n_cells == 0: print(i)
    count_all = mge_props.shape[0] / n_cells
    count_bright = mge_props[mge_props.max_intensity > thr].shape[0] / n_cells
    count_dim = mge_props[mge_props.max_intensity <= thr].shape[0] / n_cells
    # Add to dict
    roi_count_dict['mge_chan_' + str(s_ch) + '_count_all'].append(count_all)
    roi_count_dict['mge_chan_' + str(s_ch) + '_count_bright'].append(count_bright)
    roi_count_dict['mge_chan_' + str(s_ch) + '_count_dim'].append(count_dim)
    for tax in taxa_list:
        if tax != 0:
            count = hipr_props[hipr_props.max_intensity == tax].shape[0]
            roi_count_dict[str(tax)].append(count / n_cells)


# %% md

# Get correlation between species and spots

# %% codecell
l = len(roi_count_dict)
corr_mat = np.zeros((l,l))
for i, val_i in enumerate(roi_count_dict.values()):
    for j, val_j in enumerate(roi_count_dict.values()):
        # if i < j:
        vals = [np.array(v) for v in [val_i, val_j]]
        means = [np.mean(v) for v in vals]
        stds = [np.std(v) for v in vals]
        std_prod = (stds[0] * stds[1])
        if std_prod > 0:
            n = vals[0].shape[0]
            cov = np.sum((vals[0] - means[0]) * (vals[1] - means[1])) / n
            corr =  cov / std_prod
            corr_mat[i,j] = corr

# %% codecell
# Visualize correlation

plt.imshow(corr_mat, cmap='viridis')
plt.colorbar()

# %% codecell
# hierarchical clustering
import seaborn as sns

dict_labels = []
for key in roi_count_dict.keys():
    try:
        tax = int(float(key))
        lab = taxon_color_lookup.loc[taxon_color_lookup.code_b10==tax,'sci_name'].values[0]
        dict_labels.append(lab)
    except:
        dict_labels.append(key)
corr_df = pd.DataFrame(corr_mat, columns=dict_labels, index=dict_labels)
sns.clustermap(
        corr_df,
        xticklabels=1,
        yticklabels=1,
        cmap='viridis',
        cbar_kws={'label':"Pearson's correlation coefficient"}
        )
output_basename = roi_dir + '/sister_region_correlation_matrix'
ip.save_png_pdf(output_basename)

# %% codecell
# Get strong positive correlations and calculate r squred
r = 0.2
corr_df.loc[corr_df['mge_chan_' + str(s_ch) + '_count_bright'] > r, 'mge_chan_' + str(s_ch) + '_count_bright'] **2

# %% codecell
# Get strong negative correlations
r = -0.2
corr_df.loc[corr_df['mge_chan_' + str(s_ch) + '_count_bright'] < r, 'mge_chan_' + str(s_ch) + '_count_bright'] **2


# %% md

# =============================================================================
# ## Full image warping
# =============================================================================

# Started doing the manual annotation on sister regions, but realized it's easier to select sister points on the whole image

# ...Selected sister points for the whole image

# %% codecell
# Show sister points
marker_size=6

mge_pts_fn = glob.glob(config['output_dir'] + '/*pts_02.csv')[0]
hipr_pts_fn = glob.glob('../' + config_hipr['output_dir'] + '/*pts_02.csv')[0]
mge_pts = pd.read_csv(mge_pts_fn)[['X','Y']].values
hipr_pts = pd.read_csv(hipr_pts_fn)[['X','Y']].values
fig, ax, cbar = ip.plot_image(mge_cell, cmap='inferno', scalebar_resolution=mge_res)
ax.plot(mge_pts[:,0], mge_pts[:,1],'.g', ms=marker_size)
plt.sca(ax)
output_basename = config['output_dir'] + '/' + mge_sn + '_sister_pts_withraw'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()
fig, ax, cbar = ip.plot_image(hipr_raw_max_resize, cmap='inferno', scalebar_resolution=mge_res)
ax.plot(hipr_pts[:,0], hipr_pts[:,1],'.g', ms=marker_size)
plt.sca(ax)
output_basename = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_sister_pts_withraw'
ip.save_png_pdf(output_basename)
plt.show()
plt.close()

# %% md

# Warp the MGEFISH segmentation to the HiPRFISH points

# %% codecell
s_ch = 3  # Select the 633 spot channel

mge_pts = np.rint(mge_pts).astype(np.int64)
hipr_pts = np.rint(hipr_pts).astype(np.int64)
size = hipr_raw_max_resize.shape
mge_spot_seg = ip.load_output_file(
        config, 'spot_seg_max_split_fmt',
        sample_name=mge_sn, spot_chan=s_ch
        )
mge_spot = mge_raw[:,:,s_ch]
mge_spot_seg_warp = warp_image(
        mge_spot_seg.astype(np.int32),
        mge_pts,
        hipr_pts,
        size,
        dtype=np.int32,
        interpolation='nearest'
        )
mge_spot_warp = warp_image(
        mge_spot,
        mge_pts,
        hipr_pts,
        size,
        dtype=np.float64
        )
mge_spot_seg_wrp_fn = config['output_dir'] + '/' + mge_sn + '_spot_seg_warp.npy'
mge_spot_wrp_fn = config['output_dir'] + '/' + mge_sn + '_spot_raw_warp.npy'
np.save(mge_spot_seg_wrp_fn, mge_spot_seg_warp)
np.save(mge_spot_wrp_fn, mge_spot_warp)

# %% md

# Show warp on cell images

# %% codecell
hiprmge_outdir = config['output_dir'] + '/../HiPRMGEFISH'
if not os.path.exists(hiprmge_outdir): os.makedirs(hiprmge_outdir)
hiprmge_sn = '2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2'

mge_cell_warp = warp_image(
        mge_cell,
        mge_pts,
        hipr_pts,
        size,
        dtype=np.float64
        )
mge_cell_warp_fn = config['output_dir'] + '/' + mge_sn + '_cell_raw_warp.npy'
np.save(mge_cell_warp_fn, mge_cell_warp)

# %% codecell
mcs_norm = ip.zero_one_thresholding(mge_cell_warp, clims=(0.01,0.5))
plt.imshow(mcs_norm, cmap='inferno')
plt.show()
plt.close()
hms_norm = ip.zero_one_thresholding(hipr_raw_max_resize, clims=(0.01,0.3))
plt.imshow(hms_norm, cmap='inferno')
plt.show()
plt.close()
# %% codecell
overlay_shift = np.dstack([mcs_norm, hms_norm, np.zeros(hms_norm.shape)])
image_list = [mcs_norm, hms_norm, overlay_shift]
cmaps = ['inferno','inferno','']
ip.subplot_square_images(image_list,(1,3),cmaps=cmaps, scalebar_resolution=mge_res)
output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_cell_warp_overlay'
ip.save_png_pdf(output_basename)

# %% md

# Get warped spot seg regionprops

# %% codecell
mge_spot_seg_warp_props = sf.measure_regionprops(
        mge_spot_seg_warp,
        raw = mge_spot_warp
        )
mge_sswp_fn = config['output_dir'] + '/' + mge_sn + '_spot_seg_warp_props.csv'
mge_spot_seg_warp_props.to_csv(mge_sswp_fn)

# %% md

# Get hiprfish ident resized

# %% codecell
hipr_seg_bc_resize_props = sf.measure_regionprops(
        hipr_seg_resize,
        raw = hipr_bc_resize
        )
hipr_seg_bc_resize_props.to_csv(hipr_sbrp_fn)

# %% md

# Re-color the resized barcoded hiprfish image

# %% codecell
# Get the taxon legend
hipr_cell_info_fn = '../' + config_hipr['output_dir'] + '/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_3_mode_spec_cell_information_consensus.csv'
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
taxon_color_lookup['code_b10'] = [int(str(c), base=2) for c in taxon_color_lookup.code.values]
barcodes = taxon_color_lookup.loc[taxon_color_lookup.counts > 0, 'code_b10']
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
output_fn = '../' + config_hipr['output_dir'] + '/taxon_legend'
ip.save_png_pdf(output_fn)
plt.show()
plt.close()

# %% codecell
# Recolor barcoded image
barcodes_withna = np.concatenate([barcodes,np.array([35])])
cols_t20_ordered_0_withna = np.concatenate([cols_t20_ordered_0, [[0,0,1]]]).tolist()
hipr_tab20 = ip.color_bc_image(
                            hipr_bc_resize.astype(np.int64),
                            barcodes_withna,
                            cols_t20_ordered_0_withna
                            )
hipr_tab20_fn = '../' + config_hipr['output_dir'] + '/' + hipr_sn + '_ident_tab20.npy'
hipr_tab20 = hbbrr_bc_resize_tab20
np.save(hipr_tab20_fn, hipr_tab20)

# %% md

# Show overlay of spots on hiprfish ident, fiter by spot intensity

# %% codecell
im_inches=10
marker='.'
spot_col=np.array([0,1.,0,0.5])
marker_size=75
linewidths=5
ceil=0.75

mge_sswp_filt = mge_spot_seg_warp_props[mge_spot_seg_warp_props['max_intensity'] > thr]
fig, ax, cbar = ip.plot_image(hipr_tab20, scalebar_resolution=mge_res)
ref_pts = [list(c) for c in mge_sswp_filt['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
ref_pts_arr = np.array(ref_pts)
spot_int = mge_sswp_filt.max_intensity.values
spot_int /= np.max(spot_int)
spot_int[spot_int > ceil] = ceil
spot_int /= ceil
marker_size_arr = marker_size * spot_int
ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
            marker=marker, s=marker_size_arr, color=spot_col,
            linewidths=linewidths, edgecolors='none'
            )
output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_tab20_spot_overlay'
ip.save_png_pdf(output_basename)

# %% md

# =============================================================================
# ## Spot nearest neighbor analysis
# =============================================================================

# Get nearest neighbors for each spot

# %% codecell
mge_sswp_filt = mge_sswp_filt.reset_index()
hipr_seg_bc_resize_props.index
# %% codecell
knn=1

cell_coords = hipr_seg_bc_resize_props['centroid'].values
cell_coords = [list(c) for c in cell_coords]
cell_coords = np.rint(cell_coords).astype(np.int64)
# Get nearest neighbors
nbrs = NearestNeighbors(n_neighbors=knn, algorithm='kd_tree').fit(cell_coords)
spot_coords = [list(c) for c in mge_sswp_filt['centroid'].values]
spot_coords = np.rint(spot_coords).astype(np.int64)
distances, indices = nbrs.kneighbors(spot_coords)

# %% md

# Count taxon associations

# %% codecell
bc_ind_dict = {bc:i for i, bc in enumerate(barcodes_withna)}
counts = np.zeros(barcodes_withna.shape)
for i in indices:
    bc = hipr_seg_bc_resize_props.iloc[i,:]['max_intensity'].values[0]
    count_ind = bc_ind_dict[bc]
    counts[count_ind] += 1

counts
# %% md

# Get random assignment

# %% codecell
n = 10000

from random import sample
# Get total number of spots in rois
spot_intensities = mge_sswp_filt['max_intensity'].values.tolist()
# spot_intensities_full = np.zeros(hipr_seg_roi_props.shape[0])
zeros = [0]*(hipr_seg_bc_resize_props.shape[0] - len(spot_intensities))
spot_intensities_full = spot_intensities + zeros
# spot_intensities_full[:spot_intensities.shape[0]] = spot_intensities
col_list = []
sp_list = []
for i in tqdm(range(n)):
    # Assign spots randomly to cells iteratively
    spot_intensities_full_rand = sample(
                                    spot_intensities_full,
                                    len(spot_intensities_full)
                                    )
    col_list.append('spot_intensity_' + str(i))
    sp_list.append(spot_intensities_full_rand)
sp_arr = np.array(sp_list).T
sp_df = pd.DataFrame(sp_arr, columns=col_list)
hipr_seg_bc_resize_props = hipr_seg_bc_resize_props.reset_index(drop=True)
hipr_seg_bc_resize_props_randspotint = pd.concat([hipr_seg_bc_resize_props, sp_df], axis=1)
hipr_seg_bc_resize_props_randspotint.shape
# save random assignments
hsrpr_fn = hiprmge_outdir + '/' + hiprmge_sn + '_seg_bc_resized_props_randspots.csv'
hipr_seg_bc_resize_props_randspotint.to_csv(hsrpr_fn)

# %% md

# %% md

# Plot number of spots vs taxon

# %% codecell
alpha=0.5

# for each taxon
taxon_color_lookup.shape
barcodes.shape
for bc_tax in barcodes:
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
    ind_tax = np.where((hipr_seg_bc_resize_props['max_intensity'] == bc_tax).values)[0]
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
    count_ind = bc_ind_dict[bc_tax]
    true_count = counts[count_ind]
    ax.plot([true_count]*2, [0,0.5*ylims[1]], color=color)
    ax.set_xlabel('MGE association counts')
    ax.set_ylabel('Frequency')
    ax.set_title(target_taxon)
    output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_random_distribution_plot_tax_' + target_taxon
    ip.save_png_pdf(output_basename)
    plt.show()
    plt.close()

# %% md

# =============================================================================
# ## Spot region taxon association analysis
# =============================================================================

# Get region nearest neightbors

# %% codecell
# Get nearest neighbors
r_um=5  # um range

r_pix = r_um / mge_res
nbrs = NearestNeighbors(radius=r_pix, algorithm='kd_tree').fit(cell_coords)
nn_distances, nn_indices = nbrs.radius_neighbors(spot_coords)

# %% md

# Count species associations

# %% codecell
# Convert neaeres neighbro indices to taxa
dict_index_tax = {i:int(bc) for i,bc in hipr_seg_bc_resize_props['max_intensity'].iteritems()}
nn_bcs = [[dict_index_tax[i] for i in ind] for ind in nn_indices]
# Compress each list of neighbors to unique taxa
nn_bcs_unq = np.concatenate([np.unique(bcs) for bcs in nn_bcs])
# Count the number of each taxa
nn_tax_bc, nn_tax_counts = np.unique(nn_bcs_unq, return_counts=True)
nn_tax_spotfrac = nn_tax_counts / spot_coords.shape[0]

# %% md

# Count randomized species associations

# %% codecell
# Get cell nearest neighbors
cell_nn_distances, cell_nn_indices = nbrs.radius_neighbors(cell_coords)
# Convert between cell index and neighbors
dict_index_nnbc = {
        i:[dict_index_tax[j] for j in nn]
        for i, nn in zip(hipr_seg_bc_resize_props.index.values, cell_nn_indices)
        }

# %% codecell
# Randomly sample indexes of cells as proxy for spot location
n=10000
dict_tax_count_sim = defaultdict(list)
for i in tqdm(range(n)):
    spot_sample = sample(
            hipr_seg_bc_resize_props.index.values.tolist(),
            len(spot_coords)
            )
    nn_bcs_ = [dict_index_nnbc[i] for i in spot_sample]
    nn_bcs_unq_ = np.concatenate([np.unique(bcs) for bcs in nn_bcs_])
    nn_tax_bc_, nn_tax_counts_ = np.unique(nn_bcs_unq_, return_counts=True)
    for k, v in zip(nn_tax_bc_, nn_tax_counts_):
        dict_tax_count_sim[k].append(v)

dict_tax_count_sim_fn = hiprmge_outdir + '/' + hiprmge_sn + '_spot_host_association_radius_5_simulation_n_10000.yaml'
with open(dict_tax_count_sim_fn, 'w') as f:
    yaml.dump(dict_tax_count_sim, f)

# %% codecell
# calculate probability
dict_bc_sciname = {bc:sci for bc, sci in taxon_color_lookup[['code_b10','sci_name']].values}
dict_bc_prob = {}
dict_bc_ltgt = {}
for bc_tax, true_count in zip(nn_tax_bc, nn_tax_counts):
    sim_counts = np.array(dict_tax_count_sim[bc_tax])
    sim_counts_mean = np.mean(sim_counts)
    if true_count > sim_counts_mean:
        r_ = sim_counts[sim_counts >= true_count].shape[0]
        dict_bc_ltgt[bc_tax] = 1
    else:
        r_ = sim_counts[sim_counts <= true_count].shape[0]
        dict_bc_ltgt[bc_tax] = -1
    p_ = r_ / n
    try: sci_name = dict_bc_sciname[bc_tax]
    except: 'weird...'
    dict_bc_prob[bc_tax] = p_

# %% md

# Plot number of spots vs taxon

# %% codecell
alpha=1
true_col='k'
true_lw=2
lw=1
dims=(2,1)
ft=7
nbins=1000
bin_scaling=2

# for each taxon
dict_bc_tab20 = {bc:sci for bc, sci in taxon_color_lookup[['code_b10','tab20']].values}
# Recolor select barcodes
dict_bc_tab20[33] = plt.get_cmap('tab10').colors[0]  # Fretibacterium
dict_bc_tab20[39] = plt.get_cmap('tab10').colors[1]  # Butyrivibrio
dict_bc_tab20[37] = plt.get_cmap('tab10').colors[2]  # Oribacterium
for bc_tax, true_count in zip(nn_tax_bc, nn_tax_counts):
    try:
        target_taxon = dict_bc_sciname[bc_tax]
        color = dict_bc_tab20[bc_tax]
    except:
        target_taxon = 'Weird, not in probe design'
        color = (0,0,0)
    # Counts from simulation
    rand_counts = dict_tax_count_sim[bc_tax]
    # Get fraction of total spots
    rand_frac = np.array(rand_counts) / spot_coords.shape[0]
    # plot distribution of spot assignment
    nbins = np.unique(rand_frac).shape[0] * 2
    bins = np.linspace(np.min(rand_frac), np.max(rand_frac), nbins)
    # bins = np.arange(np.min(rand_frac), np.max(rand_frac))
    # hist, bin_edges = np.histogram(rand_frac)
    hist, bin_edges = np.histogram(rand_frac, bins=bins)
    x_vals = ip.get_line_histogram_x_vals(bin_edges)
    fig, ax = ip.general_plot(dims=dims, ft=ft, lw=lw)
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
    rand_count_mean = np.mean(rand_frac)
    ax.plot([rand_count_mean]*2, [0,0.75*ylims[1]], 'grey', lw=lw)
    # plot location of actual assignment number
    true_frac = true_count / spot_coords.shape[0]
    ax.plot([true_frac]*2, [0,0.75*ylims[1]], color=true_col, lw=true_lw)
    # ax.set_xlabel('MGE association counts')
    # ax.set_ylabel('Frequency')
    # ax.set_title(target_taxon)
    output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_spot_host_association_radius_5_taxon_' + target_taxon
    ip.save_png_pdf(output_basename)
    print(target_taxon)
    print('p =', dict_bc_prob[bc_tax])
    plt.show()
    plt.close()

# %% md

# =============================================================================
# ## Visualize spatial associations
# =============================================================================

# Highlight genus where spots are enriched in associations compared to random

# %% codecell
# Pick Highlight genuses
mlim=0.25  # lower limit on fraction of associated spots
plim=0.05
highlight_taxa = []
for bc_tax in nn_tax_bc:
    sim_counts = np.array(dict_tax_count_sim[bc_tax])
    mean = np.mean(sim_counts) / spot_coords.shape[0]
    p_ = dict_bc_prob[bc_tax]
    sgn = dict_bc_ltgt[bc_tax]
    if (mean > mlim) and (p_ < plim) and (sgn > 0):
        highlight_taxa.append(dict_bc_sciname[bc_tax])

highlight_taxa

# %% codecell
# Plot taxa and spots
im_inches=15  # Dimensions in inches
cell_clims=(0,0.4)
col_adj=[0.6,1,1,1]
marker='.'
spot_col= (1,1,1)
marker_size=125
linewidths=5
ceil=0.01

# Get colors
highlight_taxa = ['Oribacterium', 'Butyrivibrio', 'Christensenella']
highlighted_barcodes = [taxon_color_lookup.loc[taxon_color_lookup.sci_name==tax, 'code_b10'].values[0] for tax in highlight_taxa] # [integer,...] Barcodes to show
colors = [dict_bc_tab20[bc] for bc in highlighted_barcodes]  # [(R,G,B),...] Colors to use with each barcode
# colors = [taxon_color_lookup.loc[taxon_color_lookup.sci_name==tax, 'tab20'].values[0] for tax in highlight_taxa]  # [(R,G,B),...] Colors to use with each barcode
# adjust colors
colors = [np.array(c) * adj for c, adj in zip(colors, col_adj)]
# Get background colormap
cmap_cell_name='blacknblue'
colors_cell=[(0,0,0), plt.get_cmap('tab10').colors[0]]
n_bin=255
cmap_cell = LinearSegmentedColormap.from_list(cmap_cell_name, colors_cell, N=n_bin)
# Set up the plot
# fig = plt.figure(figsize=(dims[0], dims[1]))
# ax = plt.Axes(fig, [0., 0., 1., 1.], )
# ax.set_axis_off()
# fig.add_axes(ax)
# # Plot the grayscale image
# extent = 0, hipr_raw_max_resize.shape[1], hipr_raw_max_resize.shape[0], 0
# ax.imshow(hipr_raw_max_resize, cmap='gray', extent=extent)
fig, ax, cbar = ip.plot_image(
        hipr_raw_max_resize,
        cmap=cmap_cell,
        scalebar_resolution=mge_res,
        im_inches=im_inches,
        clims=cell_clims
        )
ax = ip.highlight_taxa(ax, hipr_bc_resize, highlighted_barcodes, colors)
# Plot the mge
ref_pts_arr = np.array(spot_coords)
spot_int = mge_sswp_filt.max_intensity.values.copy()
# spot_int /= np.max(spot_int)
spot_int[spot_int > ceil] = ceil
spot_int /= ceil
spot_col = [np.array(spot_col) * s for s in spot_int]
# marker_size = marker_size * spot_int
ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
            marker=marker, s=marker_size, color=spot_col,
            linewidths=linewidths, edgecolors='none'
            )
# mge_spot_overlay = mge_spot_warp.copy()
# mge_spot_overlay[mge_spot_overlay < 0.0015] = np.nan
# cmap_temp = copy(plt.cm.get_cmap('gray'))
# cmap_temp.set_bad(alpha = 0)
# ax.imshow(mge_spot_overlay, cmap=cmap_temp, clim=(-0.05,0.01))
# Zoom into mge fov
ax.set_xlim(shifts[1],mge_cell.shape[1] + shifts[1])
ax.set_ylim(mge_cell.shape[0] + shifts[0], shifts[0])
plt.figure(fig)
output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_spot_host_association_radius_5_seg_tab20_taxa_enriched_spot_overlay'
ip.save_png_pdf(output_basename)
plt.figure(cbar)
output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_spot_host_association_radius_5_seg_tab20_taxa_enriched_spot_overlay_eub_cbar'
ip.save_png_pdf(output_basename)


# %% codecell
dims=(0.5,1.25)
marker_size = 200
spot_col= (1,1,1)
spot_int = mge_sswp_filt.max_intensity.values.copy()
# spot_int /= np.max(spot_int)
spot_int[spot_int > ceil] = ceil
spot_int /= ceil

mx = np.max(spot_int)
mn = np.min(spot_int)
irange = [mn, np.median(spot_int), mx]
crange = [np.array(spot_col) * s for s in irange]
fig, ax = ip.general_plot(dims=dims)
ax.scatter([1,1,1],[1,2,3],marker=marker, c=crange, s=marker_size)
ax.set_axis_off()

spot_int = mge_sswp_filt.max_intensity.values.copy()
spot_int[spot_int > ceil] = ceil

mx = np.max(spot_int)
mn = np.min(spot_int)
ax.set_ylim(-1,4)
output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_spot_host_association_radius_5_seg_tab20_taxa_enriched_spot_overlay_spot_legend'
ip.save_png_pdf(output_basename)
print(mn, np.median(spot_int), mx)
































# %% md

# =============================================================================
# ## Troubleshoot
# =============================================================================


# %% codecell
# Troubleshoot sizing
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)
# %% codecell
mge_raw_bfmt = bioformats.load_image('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/data/fig_5/fig_5c/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_1_mode_airy_Airyscan_Processing_shad_stitch.czi')
mge_raw_bfmt.shape

# %% codecell
from PIL import Image
image_tif_fn = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/fig_5c/MGEFISH/2022_03_19_plaquephagelytic_sample_bmg_probes_termL2_fov_tile2_round_1_mode_airy_Airyscan_Processing_shad_stitch_new02_c3_fiji.tif'
os.path.exists(image_tif_fn)
image_pil = Image.open(image_tif_fn)
image_tif = np.array(image_pil)
image_tif.shape

# %% codecell
from skimage.registration import phase_cross_correlation
image_tif_extend = np.zeros(mge_cell.shape)
itsh = image_tif.shape
image_tif_extend[:itsh[0],:itsh[1]] = image_tif
shift_vectors = phase_cross_correlation(np.log(image_tif_extend+1), np.log(mge_cell+1))[0]
shift_vectors

# %% codecell
ip.subplot_square_images([mge_cell, image_tif_extend], (2,1), cmaps=['viridis','viridis'])

# %% md

# For some reason, javabridge is just giving the image the original shape before stitching just by adding zeros at the end.

# The coordinates for things should be the same since the upper left corner is the same.
