# %% md

# # 2022_12_19_hiprmegafish

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
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from scipy.ndimage import gaussian_filter
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans


gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2023_02_09_hiprmega/HiPR_MeGA'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %%

# Go into your configuration file and adjust all of the 'Paths' so they are relative to the workdir.

# Also adjust the 'Inputs' parameters to fit the images.

# %% codecell
config_fn = 'config.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

config_fn = '../MeGAFISH/config_mgefish.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config_mega = yaml.safe_load(f)

config_hipr_fn = '../HiPRFISH/config_hipr.yaml' # relative path to config file from workdir

with open(config_hipr_fn, 'r') as f:
    config_hipr = yaml.safe_load(f)

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%reload_ext autoreload
%autoreload 2
sys.path.append(config_mega['pipeline_path'] + '/' + config_mega['functions_path'])
sys.path
import image_plots as ip
import fn_hiprfish_classifier as fhc
import fn_spectral_images as fsi
# import segmentation_func as sf
# import spot_funcs as spf
# import fn_manual_rois as fmr
# from fn_face_morpher import warp_image, weighted_average_points

# %% md

# Get sample names

# %% codecell
input_table_mge = pd.read_csv('../MeGAFISH/' + config_mega['input_table_fn'])
sample_names_mge = input_table_mge.sample_name.values
sample_names_mge

# %% codecell
input_table_hipr = pd.read_csv('../HiPRFISH/' + config_hipr['images']['image_list_table'])
sample_names_hipr = input_table_hipr.IMAGES.values
sample_names_hipr

# %% md

# =============================================================================
# ## Find the overlay
# =============================================================================

# Run the pipeline

# %% codecell
config_fn = 'config.yaml' # relative path to config file from workdir

dry_run = False  # Just create DAG if True
n_cores = 11  # number of allowed cores for the snakemake to use
force_run = False  # Pick a rule to re-run. False if you don't want a force run.

snakefile = 'Snakefile'
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
command = " ".join(['snakemake', '-s', snakefile, '--configfile', config_fn, '-j',
                    str(n_cores), dr, fr])

run_fn = 'run_{}.sh'.format(snakefile)
with open(run_fn, 'w') as f:
    f.write(command)

print(run_fn)
print('$ ', command)


# %% md

# =============================================================================
# ## Find the overlay
# =============================================================================

# Run the pipeline

# Load test images

# %% codecell
# Pick sample
sn_mge = sample_names_mge[0]
sn_hipr = sample_names_hipr[1]
print(sn_mge)
print(sn_hipr)

# %% codecell
# Hiprfish image
hipr_max_fmt = config_hipr['output_dir'] + '/' + config_hipr['max_fmt']
hipr_max = np.load(hipr_max_fmt.format(sample_name=sn_hipr))

# %% codecell
# Mge fish image
mge_raw_fmt = config_mega['output_dir'] + '/' + config_mega['raw_fmt']
mge_raw = np.load(mge_raw_fmt.format(sample_name=sn_mge))
mge_cell = mge_raw[:,:,config_mega['cell_seg']['channels'][0]]

# %% md

# Rescale the hiprfish image

# %% codecell
mge_res = config_mega['resolution']
hipr_res = config_hipr['resolution']
factor_resize = hipr_res / mge_res
hipr_max_resize = resize(
        hipr_max,
        None,
        fx = factor_resize,
        fy = factor_resize,
        interpolation = INTER_NEAREST
        )

# %% md

# Overlay and check

# %% codecell
# Get half the difference between sizes
mshp = mge_cell.shape
hshp = hipr_max_resize.shape
shp_dff = np.abs(np.array(hshp) - np.array(mshp)) // 2
# Which is the smaller image
im_list = [mge_cell, hipr_max_resize]
i_sml = np.argmin([mshp[0],hshp[0]])
i_lrg = np.argmax([mshp[0],hshp[0]])
sml = im_list[i_sml]
lrg = im_list[i_lrg]
# Shift the smaller image so that it sits at the center of the larger image
sml_shift = np.zeros(lrg.shape)
i_shift_ends = np.array(shp_dff) + np.array(sml.shape)
sml_shift[shp_dff[0]:i_shift_ends[0], shp_dff[1]:i_shift_ends[1]] = sml

# %% codecell
# show the images
sml_norm = ip.zero_one_thresholding(sml_shift, clims=(0,0.5))
plt.imshow(sml_norm, cmap='inferno')
plt.show()
plt.close()
lrg_norm = ip.zero_one_thresholding(lrg, clims=(0,1))
plt.imshow(lrg_norm, cmap='inferno')

# %% codecell
im_inches=20
rbg_overlay = np.dstack([sml_norm, lrg_norm, np.zeros(lrg.shape)])
ip.plot_image(rbg_overlay, im_inches=im_inches, axes_off=False)

# %% md

# Run shift registration from specctral images

# %% codecell
# Run the phase cross correlation
max_shift=100
image_list = [lrg[...,None], sml_shift[...,None]]
image_registered, _, _, shift_vectors = fsi.register_shifts(image_list, max_shift=max_shift)

# %% codecell
# show the images
lrg_norm = ip.zero_one_thresholding(image_registered[:,:,0], clims=(0,1))
sml_norm = ip.zero_one_thresholding(image_registered[:,:,1], clims=(0,0.5))
# %% codecell
im_inches=20
rbg_overlay = np.dstack([sml_norm, lrg_norm, np.zeros(lrg.shape)])
ip.plot_image(rbg_overlay, im_inches=im_inches, axes_off=False)

# %% md
# check the snakemake

# %% codecell
hipr_max_resize = np.load('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_19_hiprmegafish/HiPR_MeGA/registered/HiPRFISH/2022_12_19_hiprmegafish_slide_capsB_fov_01_hipr_max_resized.npy')
mega_raw_shift = np.load('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_19_hiprmegafish/HiPR_MeGA/registered/MeGAFISH/2022_12_19_hiprmegafish_slide_capsB_fov_01_mega_raw_shift.npy')
mega_cell_shift = mega_raw_shift[:,:,0]
lrg_norm = ip.zero_one_thresholding(hipr_max_resize, clims=(0,1))
sml_norm = ip.zero_one_thresholding(mega_cell_shift, clims=(0,0.5))
im_inches=20
rbg_overlay = np.dstack([sml_norm, lrg_norm, np.zeros(lrg_norm.shape)])
ip.plot_image(rbg_overlay, im_inches=im_inches, axes_off=False)

# %% md

# Show Spots on cell ident for hipr

# %% codecell
# Plot spots on hipr
im_inches=20
marker='.'
spot_col=np.array([0,1.,0,1.])
marker_size=200
linewidths=5
ceil=0.75

mge_res = config_mega['resolution']

# Load hipr color
import pandas as pd
hipr_col_fmt = config['output_dir'] + '/' + config['hipr']['seg_col_resize']
mega_props_fmt = config['output_dir'] + '/' + config['mega']['spot_props_shift']
overlay_fmt = config['output_dir'] + '/' + config['hipr_mega']['overlay_plot']
overlay_dir = os.path.split(overlay_fmt)[0]
if not os.path.exists(overlay_dir): os.makedirs(overlay_dir)
for sn in tqdm(sample_names_mge):
    for c_ch in config_mega['cell_seg']['channels']:
        for s_ch in config_mega['spot_seg']['channels']:
            hipr_col = np.load(hipr_col_fmt.format(sample_name=sn))
            # Load spot seg properties
            props_fn = mega_props_fmt.format(sample_name=sn, cell_chan=c_ch, spot_chan=s_ch)
            spot_props_shift = pd.read_csv(props_fn)
            # Plot col
            fig, ax, cbar = ip.plot_image(hipr_col, scalebar_resolution=mge_res, im_inches=im_inches)
            # Plot spots
            ref_pts = [eval(c) for c in spot_props_shift['centroid'].values]
            ref_pts = np.rint(ref_pts).astype(np.int64)
            ref_pts_arr = np.array(ref_pts)
            spot_int = spot_props_shift.max_intensity.values
            spot_int /= np.max(spot_int)
            spot_int[spot_int > ceil] = ceil
            spot_int /= ceil
            marker_size_arr = marker_size * spot_int
            ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
                        marker=marker, s=marker_size_arr, color=spot_col,
                        linewidths=linewidths, edgecolors='none'
                        )
            out_fn = overlay_fmt.format(sample_name=sn, cell_chan=c_ch, spot_chan=s_ch)
            out_fn = os.path.splitext(out_fn)[0]
            ip.save_png_pdf(out_fn)
            print(sn)
            plt.show()
            plt.close()
# output_basename = hiprmge_outdir + '/' + hiprmge_sn + '_tab20_spot_overlay'
# ip.save_png_pdf(output_basename)

# %% codecell
hipr_seg = np.load('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_19_hiprmegafish/HiPRFISH/segmentation/2022_12_19_hiprmegafish_slide_capsB_fov_01_seg.npy')

fig, ax, cbar = ip.plot_image(ip.seg2rgb(hipr_seg), scalebar_resolution=mge_res, im_inches=im_inches)


# %% md

# Choose new colors for hiprfish barcodes

# %% codecell
# Load all cell props classsif
hipr_imtab_fn = config['hipr_dir'] + '/' + config_hipr['images']['image_list_table']
hipr_imtab = pd.read_csv(hipr_imtab_fn)
hipr_sample_names = hipr_imtab['IMAGES']
fmt = config['hipr_dir'] + '/' + config_hipr['output_dir'] + '/' + config_hipr['classif_filt_fmt']
props_combo = pd.DataFrame()
for sn in hipr_sample_names:
    fn = fmt.format(sample_name = sn)
    p = pd.read_csv(fn)
    props_combo = props_combo.append(p)
# %% codecell
# get counts on barcodes
bc_counts = props_combo['cell_barcode'].value_counts()
# %% codecell
# Order barcodes
barcodes = bc_counts.index.astype(np.int32).tolist()
missing_bcs = [11, 1011]
barcodes += missing_bcs
# %% codecell
# apply tab20 colors
colors = list(plt.get_cmap('tab20').colors)
colors[4:6]
del colors[4:6]
colors = colors[:len(barcodes)]
df_recolor = pd.DataFrame({'barcode':barcodes, 'color':colors})
# %% codecell
# Save new tab20 color csv
out_fn = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR'] + '/welch2016_5b_no_633_channel_colors_20230125.csv'
df_recolor.to_csv(out_fn, index=False)
# %% codecell
plt.scatter(np.arange(len(colors)), np.ones(len(colors)), color=colors)

# %% md

# subset seg for classif spectra test

# %% codecell
hipr_seg = np.load('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_19_hiprmegafish/HiPRFISH/segmentation/2022_12_19_hiprmegafish_slide_capsB_fov_01_seg.npy')
hipr_reg = np.load('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_19_hiprmegafish/HiPRFISH/prep/2022_12_19_hiprmegafish_slide_capsB_fov_01_registered.npy')
hipr_props = pd.read_csv('/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_19_hiprmegafish/HiPRFISH/seg_props/2022_12_19_hiprmegafish_slide_capsB_fov_01_seg_props.csv')
# %% codecell
seg_sub = hipr_seg
labels_sub = np.unique(seg_sub)
labels_sub.shape
# %% codecell
reg_sub = hipr_reg[2000:2100,2000:2100,:]
props_sub = hipr_props[hipr_props['label'].isin(labels_sub)]
props_sub.shape
# %% codecell
dir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2022_12_19_hiprmegafish/HiPRFISH/classification/test'
np.save(dir + '/seg_sub_' + str(labels_sub.shape[0]) + '.npy', seg_sub)
np.save(dir + '/reg_sub_' + str(labels_sub.shape[0]) + '.npy', reg_sub)
props_sub.to_csv(dir + '/props_sub_' + str(labels_sub.shape[0]) + '.csv', index=False)

# %% codecell
# timing results
n_cells = [22,129, 1531, 13903]

x = np.array([n_cells for i in range(4)]).T

t1 = [0.012744665145874023,0.013984203338623047,0.013759613037109375,0.012859106063842773]
t2 = [1.4707005023956299,1.7331557273864746, 4.193493843078613,23.198103189468384]
t3 = [0.0009152889251708984,0.0011816024780273438,0.0031037330627441406,0.018268823623657227]
t4 = [0.0011217594146728516,0.0054149627685546875,0.061231374740600586,0.5489354133605957]
y = np.array([t1, t2, t3, t4]).T
plt.plot(x, y, label=['1','2','3','4'])
plt.legend()
# %% md


# =============================================================================
# ## Look at lautropia spectra I know are wrong
# =============================================================================

# %% codecell
# Load resized/shifted: hipr reg image, mega raw image, hipr seg image, hipr classif props
sn = '2022_12_19_hiprmegafish_slide_capsB_fov_01'
mega_out_dir = config['mega_dir'] + '/' + config_mega['output_dir']
cell_fn = mega_out_dir + '/' + config_mega['raw_fmt'].format(sample_name=sn)
mega_raw = np.load(cell_fn)
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
sum_fn = hipr_out_dir + '/' + config_hipr['sum_fmt'].format(sample_name=sn)
sum_im = np.load(sum_fn)
reg_fn = hipr_out_dir + '/' + config_hipr['reg_fmt'].format(sample_name=sn)
reg = np.load(reg_fn)
seg_fn = hipr_out_dir + '/' + config_hipr['seg_fmt'].format(sample_name=sn)
seg = np.load(seg_fn)
props_fn = hipr_out_dir + '/' + config_hipr['props_classif_fmt'].format(sample_name=sn)
props = pd.read_csv(props_fn)


# %% codecell
# show mega raw cell image
clims = ['min',5]
im_inches=10
# cell_im = mega_raw[:,:,config_mega['cell_seg']['channels'][0]]
ip.plot_image(sum_im, im_inches=im_inches, cmap='inferno', axes_off=False, clims=clims)
plt.show()

# %% codecell
# Pick region to subset
i = [250,0]
w = [400,300]
sum_sub = sum_im[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
ip.plot_image(sum_sub, cmap='inferno', clims=clims)

# %% codecell
# subset all images
seg_sub = seg[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
reg_sub = reg[i[0]:i[0]+w[0], i[1]:i[1]+w[1], :]
ip.plot_image(seg_sub)

# %% codecell
# Subset the seg props
cid_list = np.unique(seg_sub[seg_sub > 0])
props_filt = props.loc[props['label'].isin(cid_list), :]
props_filt.shape

# %% codecell
# Show the cell spectra
avgint_cols = [str(i) for i in range(config_hipr['chan_start'],config_hipr['chan_end'])]
for cb in props_filt['cell_barcode'].unique():
    p = props_filt.loc[props_filt['cell_barcode'] == cb, avgint_cols]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, p, {'lw':1,'alpha':0.9,'color':'r'})
    ax.set_title(cb)

# %% codecell
# Get  spectra from e coli reference
def _add_rough_class_columns(spectra, code, params):
    code = code.zfill(5)
    numeric_code_list = [int(c) for c in list(code)]
    numeric_code_list.insert(2,0)  # ran as 5b classifier, need to add r6 and r7 zeros
    numeric_code_list.insert(2,0)
    nc_arr = np.array(numeric_code_list)
    shp = spectra.shape[0]
    for ch in range(len(params['rough_classifier_matrix'])):
        # If the code should not show up in this laser, or you are simulating negative data, give a zero
        bool = any(nc_arr * np.array(params['rough_classifier_matrix'][ch]))
        if bool:
            col = np.ones((shp ,1), dtype=int)
        # If the code should show up in this laser, give a one
        else:
            col = np.zeros((shp ,1), dtype=int)
        spectra = np.hstack([spectra, col])
    return(spectra, numeric_code_list)

hipr_ref_dir = config['hipr_dir'] + '/' + config_hipr['hipr_ref_dir']
ref_fmt = hipr_ref_dir + '/' + config_hipr['ref_files_fmt']
barcodes = props_filt['cell_barcode'].unique()
params = fhc.get_7b_params()
n_las=4
ref_spec = []
for bc in barcodes:
    bc = str(int(bc))
    bc10 = int(bc, 2)
    ref_fn = ref_fmt.format(bc10)
    rsi = params['ref_spec_indices']
    rs = pd.read_csv(ref_fn).values[:,rsi[0]:rsi[1]]
    rs, ncl = _add_rough_class_columns(rs, bc, params)
    code = ''.join([str(c) for c in ncl])
    rs = np.hstack([rs, np.repeat(code, rs.shape[0])[:,None]])
    ref_spec.append(rs)
ref_spec = np.vstack(ref_spec)
ref_spec.shape

# %% codecell
# Get the avg spectra from the props
avgint_cols = [str(i) for i in range(config_hipr['chan_start'],config_hipr['chan_end'])]
props_filt.columns
cell_spec = props_filt[avgint_cols].values
cell_spec = np.hstack([cell_spec, np.zeros((cell_spec.shape[0], 6))]) # Add zeros for red channel
cell_spec = np.hstack([cell_spec, props_filt[['rough_class_0', 'rough_class_1', 'rough_class_2']]])
cell_spec = np.hstack([cell_spec, np.zeros((cell_spec.shape[0], 1))])  # Rough classifier for red channel
cell_spec = np.hstack([cell_spec, props_filt[['cell_barcode']]])  # Rough classifier for red channel
cell_spec.shape


# %% codecell
# get  simulation training data
ref_train_dir = config['hipr_dir'] + '/' + config_hipr['reference_training']['out_dir']
td_fn = ref_train_dir + '/PROBEDESIGN_welch2016_BITS_7_NSIMS_2000_OBJ_training_data.csv'
tdn_fn = ref_train_dir + '/PROBEDESIGN_welch2016_BITS_7_NSIMS_2000_OBJ_training_data_neg.csv'
training_data = pd.read_csv(td_fn).values
training_data_neg = pd.read_csv(tdn_fn).values
training_data_full = np.vstack([training_data, training_data_neg])
training_data_full.shape

# %% codecell
# run the umap transform of the cell spectra, the simulation spectra, and the reference spectra
n_chan = 4
nc = 1 + n_chan
tdf_arr = training_data_full[:,:-nc]
scaler = StandardScaler().fit(tdf_arr)
tdf_scaled = scaler.transform(tdf_arr)
tdfs_fit = tdf_scaled
# tdfs_fit = np.hstack([tdf_scaled, training_data_full[:,-nc:-1]])
tdf_y = [re.sub('_error','.404', str(e)) for e in training_data_full[:,-1]]
umap_obj = UMAP(n_neighbors=25, metric=fhc.channel_cosine_intensity_7b_v2)
umap_transform = umap_obj.fit(tdfs_fit, y=tdf_y)
ut_trn = umap_transform.embedding_

# %% code_col
rs = ref_spec[:,:-nc].astype(np.float64)
cs = cell_spec[:,:-nc].astype(np.float64)
rs_norm = rs / np.max(rs, axis=1)[:,None]
cs_norm = cs / np.max(cs, axis=1)[:,None]
rs_scaled = scaler.transform(rs_norm)
cs_scaled = scaler.transform(cs_norm)
ut_ref = umap_transform.transform(rs_scaled)
ut_cell = umap_transform.transform(cs_scaled)
ut_trn_ref_cell = np.vstack([ut_trn, ut_ref, ut_cell])
ut_trn_ref_cell.shape



# %% codecell
# Set up labeles for the umap
labels_trn = [str(e) for e in training_data_full[:,-1]]
labels_ref = [str(e) + '_ref' for e in ref_spec[:,-1]]
labels_cell = [str(e) + '_cell' for e in cell_spec[:,-1]]
labels_trn_ref_cell = labels_trn + labels_ref + labels_cell

# %% codecell
# Show the trainig umap projections
dims=(10,10)
ms=1
ft=14
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_trn, labels_trn, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft)

# %% codecell
# Show the reference umap projections
dims=(10,10)
ms=10
ft=14
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_ref, labels_ref, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft)

# %% codecell
# Show the cell average  umap projections
dims=(10,10)
ms=10
ft=14
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_cell, labels_cell, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft)

# %% codecell
# Show all the umap projections
dims=(10,10)
ms=10
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_trn_ref_cell, labels_trn_ref_cell, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% md
# =============================================================================
# ## run the umap transform including the rough classifier
# =============================================================================

# %% codecell
# run the umap transform including the rough classifier
tdfs_fit_rc = np.hstack([tdf_scaled, training_data_full[:,-nc:-1]])
umap_obj_rc = UMAP(n_neighbors=25, metric=fhc.channel_cosine_intensity_7b_v2)
umap_transform_rc = umap_obj_rc.fit(tdfs_fit_rc, y=tdf_y)
ut_trn_rc = umap_transform_rc.embedding_

# %% codecell
rs_fit_rc = np.hstack([rs_scaled, ref_spec[:,-nc:-1]])
cs_fit_rc = np.hstack([cs_scaled, cell_spec[:,-nc:-1]])
ut_ref_rc = umap_transform_rc.transform(rs_fit_rc)
ut_cell_rc = umap_transform_rc.transform(cs_fit_rc)
ut_trn_ref_cell_rc = np.vstack([ut_trn_rc, ut_ref_rc, ut_cell_rc])
ut_trn_ref_cell_rc.shape

# %% codecell
# Show the trainig umap projections witht the RC
dims=(10,10)
ms=1
ft=14
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_trn_rc, labels_trn, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft)

# %% codecell
# Show the reference umap projections with the rough classifier
dims=(10,10)
ms=10
ft=14
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_ref_rc, labels_ref, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft)

# %% codecell
# Show the cell average  umap projections
dims=(10,10)
ms=10
ft=14
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_cell_rc, labels_cell, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft)

# %% codecell
# Show all the umap projections
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_trn_ref_cell_rc, labels_trn_ref_cell, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% md
# =============================================================================
# ## Try just projecting the pixel spectra onto a umap
# =============================================================================

# %% codecell
# Get the pixel by pixel spectra for the cell seg
mask_sub = seg_sub > 0
spec_pix = reg_sub[mask_sub, :]
spec_pix_norm = spec_pix / np.max(spec_pix, axis=1)[:,None]
scaler_pix = StandardScaler().fit(spec_pix_norm)
spec_pix_norm_scale = scaler_pix.transform(spec_pix_norm)

# %% codecell
# run the umap on the pixel spectra
umap_obj_pix = UMAP(n_neighbors=25, metric=fhc.channel_cosine_intensity_7b_v2)
umap_transform_pix = umap_obj_pix.fit(spec_pix_norm)
ut_pix = umap_transform_pix.embedding_
labels_pix = np.repeat('pix', ut_pix.shape[0])

# %% codecell
# Plot the umap on pixel spectra, simulation, and reference
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, _ = fhc.plot_umap(ut_pix, labels_pix, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)


# %% codecell
# Cluster the spectra
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(spec_pix_norm_scale)

# %% codecell
# show the clusters
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_pix, labels_clust, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# clustering with optics
from sklearn.cluster import OPTICS

clust = OPTICS(min_samples=100, xi=0.05, min_cluster_size=0.01, metric=fhc.channel_cosine_intensity_7b_v2)
clust.fit(spec_pix_norm_scale)
labels_optics = clust.labels_

# %% codecell
# show the optics clusters
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_pix, labels_optics, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# try clustering on pca space
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(spec_pix_norm_scale)
pca.explained_variance_ratio_

# %% codecell
# Try clustering on umap space
kmeans = KMeans(n_clusters=2, random_state=42)
labels_ut_kmeans = kmeans.fit_predict(ut_pix)
# %% codecell
# show the  clusters
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_pix, labels_ut_kmeans, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)


# %% codecell
# try dbscan on umap reduction
clust = OPTICS(min_samples=100, xi=0.05, min_cluster_size=0.01, metric=fhc.channel_cosine_intensity_7b_v2)
clust.fit(ut_pix)
labels_ut_optics = clust.labels_
# %% codecell
# show the  clusters
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax = fhc.plot_umap(ut_pix, labels_ut_optics, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# Try hdbscan
labels_ut_hdbscan = HDBSCAN(min_cluster_size=20, min_samples=100).fit_predict(ut_pix)
# %% codecell
# show the  clusters
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_pix, labels_ut_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

col_df

# %% codecell
# Re project the pixels onto the image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
col_dict = {l:c for l, c in col_df[['numeric_barcode','color']].values}

sum_sub_norm = sum_sub.copy()
sum_sub_norm[sum_sub_norm > clims[1]] = clims[1]
sum_sub_norm = sum_sub_norm / clims[1]
im_clust = np.zeros(sum_sub.shape + (len(col_df.color.values[0]),))
im_clust[:,:,3] = 1
for lab, ind in zip(labels_clust, pix_ind):
    x, y = ind
    col = np.array(col_dict[lab])
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = col
# %% codecell
ip.plot_image(sum_sub, cmap='inferno', clims=clims)
plt.show()
plt.close()
ip.plot_image(im_clust)

# %% codecell
# Re project the pixels onto the image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
col_dict = {l:c for l, c in col_df[['numeric_barcode','color']].values}

sum_sub_norm = sum_sub.copy()
sum_sub_norm[sum_sub_norm > clims[1]] = clims[1]
sum_sub_norm = sum_sub_norm / clims[1]
im_clust = np.zeros(sum_sub.shape + (len(col_df.color.values[0]),))
im_clust[:,:,3] = 1
for lab, ind in zip(labels_kmeans, pix_ind):
    x, y = ind
    col = np.array(col_dict[lab])
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = col
# %% codecell
ip.plot_image(sum_sub, cmap='inferno', clims=clims)
plt.show()
plt.close()
ip.plot_image(im_clust)


# %% md
# =============================================================================
# just look at the spectra from the upper left region of the subset image
# =============================================================================

# %% codecell
reg_ul = reg_sub[:50,:50,:]
seg_ul = seg_sub[:50,:50]
sum_ul = sum_sub[:50,:50]
spec_ul = reg_ul[seg_ul > 0]
ip.plot_image(sum_ul, cmap='inferno', clims=clims)
plt.show()
plt.close()
ip.plot_image(seg_ul)
plt.show()
plt.close()
reg_lr = reg_sub[50:,50:,:]
seg_lr = seg_sub[50:,50:]
sum_lr = sum_sub[50:,50:]
spec_lr = reg_lr[seg_lr > 0]
ip.plot_image(sum_lr, cmap='inferno', clims=clims)
plt.show()
plt.close()
ip.plot_image(seg_lr)
plt.show()
plt.close()


# %% codecell
# just look at the spectra from the lower right region
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spec_ul, {'lw':0.1,'alpha':0.5,'color':'r'})
ax.set_title('upperleft')
plt.show()
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spec_lr, {'lw':0.1,'alpha':0.5,'color':'r'})
ax.set_title('lowerright')

# %% codecell
# Run gaussian blurring on the reg image

sigma=1.4
reg_sub_smooth = np.empty(reg_sub.shape)
for i in range(reg_sub.shape[2]):
    reg_sub_smooth[:,:,i] = gaussian_filter(reg_sub[:,:,i], sigma=sigma)

# %% codecell
# now show images again
reg_ul = reg_sub_smooth[:50,:50,:]
sum_ul = np.sum(reg_ul, axis=2)
spec_ul = reg_ul[seg_ul > 0]
ip.plot_image(sum_ul, cmap='inferno', clims=clims)
plt.show()
plt.close()
ip.plot_image(seg_ul)
plt.show()
plt.close()
reg_lr = reg_sub_smooth[50:,50:,:]
sum_lr = np.sum(reg_lr, axis=2)
spec_lr = reg_lr[seg_lr > 0]
ip.plot_image(sum_lr, cmap='inferno', clims=clims)
plt.show()
plt.close()
ip.plot_image(seg_lr)
plt.show()
plt.close()

# %% codecell
# Show spectra again
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spec_ul, {'lw':0.1,'alpha':0.5,'color':'r'})
ax.set_title('upperleft')
plt.show()
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spec_lr, {'lw':0.1,'alpha':0.5,'color':'r'})
ax.set_title('lowerright')

# %% codecell
# run the umap on the smoothed pixel
spec_pix_smth = reg_sub_smooth[seg_sub > 0]
sps_norm = spec_pix_smth / np.max(spec_pix_smth, axis=1)[:,None]
sps_scaler = StandardScaler().fit(sps_norm)
sps_scale = sps_scaler.transform(sps_norm)
umap_obj_pix_smth = UMAP(n_neighbors=25, metric=fhc.channel_cosine_intensity_7b_v2)
umap_transform_pix_smth = umap_obj_pix_smth.fit(sps_scale)
ut_pixs_smth = umap_transform_pix_smth.embedding_
labels_pix = np.repeat('pix', ut_pixs_smth.shape[0])
# %% codecell
# Plot the umap on pixel spectra
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, _ = fhc.plot_umap(ut_pixs_smth, labels_pix, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# cluster the smoothed spectra
labels_ut_smth_hdbscan = HDBSCAN(min_cluster_size=100, min_samples=250).fit_predict(ut_pixs_smth)

# %% codecell
# Plot the umap on pixel spectra
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_pixs_smth, labels_ut_smth_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# Re project the pixels onto the image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask_sub)
# for each index draw the color
col_dict = {l:c for l, c in col_df[['numeric_barcode','color']].values}

sum_sub_norm = sum_sub.copy()
sum_sub_norm[sum_sub_norm > clims[1]] = clims[1]
sum_sub_norm = sum_sub_norm / clims[1]
im_clust = np.zeros(sum_sub.shape + (len(col_df.color.values[0]),))
im_clust[:,:,3] = 1
for lab, ind in zip(labels_ut_smth_hdbscan, pix_ind):
    x, y = ind
    col = np.array(col_dict[lab])
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = col
# %% codecell
ip.plot_image(sum_sub, cmap='inferno', clims=clims)
plt.show()
plt.close()
ip.plot_image(im_clust)


# %% md
# =============================================================================
# ## Try doing the clustering with the ground truth spectra
# =============================================================================

# %% codecell
# Get the 18 spectra
hipr_ref_dir = config['hipr_dir'] + '/' + config_hipr['hipr_ref_dir']
ref_fmt = hipr_ref_dir + '/' + config_hipr['ref_files_fmt']
probe_design_dir = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config_hipr['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
barcodes = probe_design['code'].unique()
barcodes
# %% codecell
params = fhc.get_7b_params()
n_las=4
ref_spec = []
for bc in barcodes:
    bc = str(int(bc))
    bc10 = int(bc, 2)
    ref_fn = ref_fmt.format(bc10)
    rsi = params['ref_spec_indices']
    rs = pd.read_csv(ref_fn).values[:,rsi[0]:rsi[1]]
    # rs, ncl = _add_rough_class_columns(rs, bc, params)
    # code = ''.join([str(c) for c in ncl])
    rs = np.hstack([rs, np.repeat(int(bc), rs.shape[0])[:,None]])
    ref_spec.append(rs)
ref_spec = np.vstack(ref_spec)
ref_spec.shape


# %% codecell
# NOrmalize
rs_norm = ref_spec / np.max(ref_spec[:,:-1], axis=1)[:,None]
# scale
rs_scaler = StandardScaler().fit(rs_norm)
rs_scale = rs_scaler.transform(rs_norm)
# %% codecell
# umap reduction
umap_ref = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_7b_v2)
umap_transform_ref = umap_ref.fit(rs_scale)
ut_ref = umap_transform_ref.embedding_
labels_ut_ref = ref_spec[:,-1]

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_ref, labels_ut_ref, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# cluster
labels_ut_ref_hdbscan = HDBSCAN(min_cluster_size=20, min_samples=100).fit_predict(rs_scale)

# %% codecell
# Plot the umap
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_ref, labels_ut_ref_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% md

# Show cell average spectra

# %% codecell
# Separate upper left and lower right cells
lab_ul = np.unique(seg_ul)
lab_lr = np.unique(seg_lr)
cells_ul = props.loc[props['label'].isin(lab_ul), avgint_cols].values
cells_lr = props.loc[props['label'].isin(lab_lr), avgint_cols].values
cell_spec = np.vstack([cells_ul, cells_lr])
labels_cell = ['ul']*cells_ul.shape[0] + ['lr']*cells_lr.shape[0]

# %% codecell
# NOrmalize
cs_norm = cell_spec / np.max(cell_spec, axis=1)[:,None]
# scale
cs_scaler = StandardScaler().fit(cs_norm)
cs_scale = cs_scaler.transform(cs_norm)
# %% codecell
# umap reduction
umap_cell = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_7b_v2)
umap_transform_cell = umap_cell.fit(cs_scale)
ut_cell = umap_transform_cell.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(5,5)
ms=100
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell, labels_cell, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% md

################
# Now combine cells with reference spectra
################

# %% codecell
# NOrmalize
# avgint_cols = [str(i) for i in range(config_hipr['chan_start'],config_hipr['chan_end'])]
# cell_spec = np.hstack([props_filt[avgint_cols], np.zeros((cell_spec.shape[0], 6))])  # Add 633 channel
ch633 = 6   # Remove 633 channel
rs_norm_ = rs_norm[:,:-(1+ch633)]
rcs_norm = np.vstack([rs_norm_, cs_norm])
# scale
rcs_scaler = StandardScaler().fit(rcs_norm)
rcs_scale = rcs_scaler.transform(rcs_norm)
# %% codecell
# umap reduction
umap_ref_cell = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_ref_cell = umap_ref_cell.fit(rcs_scale)
ut_ref_cell = umap_transform_ref_cell.embedding_
labels_ut_ref_cell = ref_spec[:,-1].astype(str).tolist() + labels_cell

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_ref_cell, labels_ut_ref_cell, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# cluster
labels_ut_ref_cell_hdbscan = HDBSCAN(min_cluster_size=5, min_samples=100).fit_predict(rcs_scale)

# %% codecell
# Plot the umap
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_ref_cell, labels_ut_ref_cell_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)




# %% md

# %% md
# =============================================================================
# ## Try projecting all the cell spectra onto umap
# =============================================================================

# %% codecell
cell_spec_full = props[avgint_cols].values
labels_cell_full = [0]*cell_spec_full.shape[0]
cell_spec_full.shape

# %% codecell
# NOrmalize
csf_norm = cell_spec_full / np.max(cell_spec_full, axis=1)[:,None]
# scale
csf_scaler = StandardScaler().fit(csf_norm)
csf_scale = csf_scaler.transform(csf_norm)
# %% codecell
# umap reduction
umap_cell_full = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_cell_full = umap_cell_full.fit(csf_scale)
ut_cell_full = umap_transform_cell_full.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell_full, labels_cell_full, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# cluster
labels_utcsf_hdbscan = HDBSCAN(min_cluster_size=10, min_samples=23).fit_predict(ut_cell_full)
# labels_csf_hdbscan = HDBSCAN(min_cluster_size=10, min_samples=10).fit_predict(cell_spec_full)

# %% codecell
# Plot the umap
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell_full, labels_utcsf_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% md

############
# all cell average spectra With reference spectra

# %% codecell
rcs_norm = np.vstack([rs_norm_, csf_norm])
# scale
rcs_scaler = StandardScaler().fit(rcs_norm)
rcs_scale = rcs_scaler.transform(rcs_norm)
# %% codecell
# umap reduction
umap_ref_cell = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_ref_cell = umap_ref_cell.fit(rcs_scale)
ut_ref_cell = umap_transform_ref_cell.embedding_
labels_ut_ref_cell = ref_spec[:,-1].tolist() + labels_cell_full

# %% codecell
# plot with labels
dims=(7,7)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_ref_cell, labels_ut_ref_cell, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% md

# different umaps

# %% codecell
# ECS distance
umap_cell_full = UMAP(n_neighbors=10, metric=fhc.euclid_dist_cumul_spec)
umap_transform_cell_full = umap_cell_full.fit(csf_scale)
ut_cell_full = umap_transform_cell_full.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell_full, labels_cell_full, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# raw spectra
umap_cell_full = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_cell_full = umap_cell_full.fit(cell_spec_full)
ut_cell_full = umap_transform_cell_full.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell_full, labels_cell_full, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# ecs with raw spectra
umap_cell_full = UMAP(n_neighbors=10, metric=fhc.euclid_dist_cumul_spec)
umap_transform_cell_full = umap_cell_full.fit(cell_spec_full)
ut_cell_full = umap_transform_cell_full.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell_full, labels_cell_full, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% md

############
# subset pixel spectra With reference spectra

# %% codecell
rps_norm = np.vstack([rs_norm_, sps_norm])
# scale
rps_scaler = StandardScaler().fit(rps_norm)
rps_scale = rps_scaler.transform(rps_norm)
# %% codecell
# umap reduction
umap_ref_pix = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_ref_pix = umap_ref_pix.fit(rps_scale)
ut_ref_pix = umap_transform_ref_pix.embedding_
labels_ut_ref_pix = ref_spec[:,-1].tolist() + labels_ut_smth_hdbscan.tolist()

# %% codecell
# plot with labels
dims=(7,7)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_ref_pix, labels_ut_ref_pix, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)


# %% md
# =============================================================================
# ## Plot raw spectra
# =============================================================================

# %% codecell
# Get the 15 spectra from e coli reference
hipr_ref_dir = config['hipr_dir'] + '/' + config_hipr['hipr_ref_dir']
ref_fmt = hipr_ref_dir + '/' + config_hipr['ref_files_fmt']
probe_design_dir = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config_hipr['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
barcodes = probe_design['code'].unique()
barcodes
# %% codecell
def from_7bit_to_10bit_code(code):
    code = str(int(code))
    code = code.zfill(7)
    numeric_code_list = [int(c) for c in list(code)]
    for i in range(2):
        numeric_code_list.insert(4,0)  # ran as 5b classifier, need to add r4,5 zeros
    numeric_code_list.insert(1,0)  # Add r9 zero
    return ''.join([str(c) for c in numeric_code_list])

# %% codecell
params = fhc.get_7b_params()
n_las=4
ref_spec = []
training_out_dir = config['hipr_dir'] + '/' + config_hipr['reference_training']['out_dir']
for bc in barcodes:
    bc_ = from_7bit_to_10bit_code(bc)
    print(bc,bc_)
    bc10 = int(bc_, 2)
    ref_fn = ref_fmt.format(bc10)
    rsi = params['ref_spec_indices']
    rs = pd.read_csv(ref_fn).values[:,rsi[0]:rsi[1]-6]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, rs, {'lw':0.1,'alpha':0.5,'color':'r'})
    ax.set_title(bc_)
    out_bn = training_out_dir + '/reference_spectra_code_' + bc_
    ip.save_png_pdf(out_bn)
    plt.show()
    # rs, ncl = _add_rough_class_columns(rs, bc, params)
    # code = ''.join([str(c) for c in ncl])

# %% md

# PLot the spectra from the cells as grouped by the kmeans clustering

# %% codecell
n_bc = barcodes.shape[0]
kmeans = KMeans(n_clusters=n_bc, random_state=42)
labels_csf_kmeans = kmeans.fit_predict(csf_scale)
clust_assgn = pd.DataFrame({'label':props.label.values,'kmeans_cluster':labels_csf_kmeans})
# clust_assgn.to_csv(classif_out_dir + '/cell_cluster_assignment.csv')

# %% codecell
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
classif_out_dir = hipr_out_dir + '/kmeans_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_csf_kmeans)
for l in labs:
    bool = (labels_csf_kmeans == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_cluster_' + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# Try some  normalization
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
classif_out_dir = hipr_out_dir + '/kmeans_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_csf_kmeans)
for l in labs:
    bool = (labels_csf_kmeans == l)
    spec_ = cell_spec_full[bool,:]
    spec_norm = spec_ / np.max(spec_, axis=1)[:,None]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_norm, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_norm_cluster_' + str(l).zfill(2)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# Try row normalization combined with laser normalization
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
classif_out_dir = hipr_out_dir + '/kmeans_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_csf_kmeans)
cell_spec_full_norm = cell_spec_full / np.max(cell_spec_full, axis=1)[:,None]
c = params['channel_indices']
las_max = [np.mean(cell_spec_full_norm[:,c[i]:c[i+1]]) for i in range(3)]
for l in labs:
    bool = (labels_csf_kmeans == l)
    spec_ = cell_spec_full_norm[bool,:]
    for i in range(3):
        spec_[:,c[i]:c[i+1]] = spec_[:,c[i]:c[i+1]] / las_max[i]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_weird_norm_cluster_' + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% md
# =============================================================================
# ## manual classification
# =============================================================================

# %% codecell
# Dict of assignemnts 7bit barcode: clusters
dict_code_clust = {
        '1100100':[9,7,5],
        '0100000':[1,13],
        '0100001':[3],
        '1000000':[2],
        '1000101':[12],
        '1000110':[14,17,8],
        '1100001':[10],
        '1100010':[15,16,0],
        '1100111':[11,6],
        '1000011':[4]
}

certain = [4,11,15,16,10,12,2,3,9,12,7] # which clusters are strongly ID'd

# %% codecell
# add new classification to props
props_reclass = pd.DataFrame([])
for bc, clsts in dict_code_clust.items():
    for cl in clsts:
        p_ = props.copy()[labels_csf_kmeans == cl]
        p_['code_manual'] = bc
        crtn = 1 if cl in certain else 0
        p_['manual_confidence'] = crtn
        p_['kmeans_cluster'] = cl
        props_reclass = props_reclass.append(p_)

props_reclass.columns

# %% md

# choose new colors for barcodes

# %% codecell
# get counts on barcodes
bc_counts = props_reclass['code_manual'].value_counts()
# Order barcodes
barcodes = bc_counts.index.astype(int).tolist()

# %% codecell
# add missing barcodes
hipr_ref_dir = config['hipr_dir'] + '/' + config_hipr['hipr_ref_dir']
ref_fmt = hipr_ref_dir + '/' + config_hipr['ref_files_fmt']
probe_design_dir = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config_hipr['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
barcodes_all = probe_design['code'].unique()
missing_bcs = [int(bc) for bc in barcodes_all if bc not in barcodes]
barcodes += missing_bcs
barcodes

# %% codecell
# apply tab20 colors
colors = list(plt.get_cmap('tab20').colors)
colors[4:6]
del colors[4:6]  # Delete greens
colors = colors[:len(barcodes)]
df_bc_col = pd.DataFrame({'barcode_7b':barcodes, 'color':colors})
df_bc_col

# %% codecell
# Save new tab20 color csv
out_fn = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR'] + '/welch2016_5b_no_633_channel_colors_20230202.csv'
df_bc_col.to_csv(out_fn, index=False)
# %% codecell
plt.scatter(np.arange(len(colors)), np.ones(len(colors)), color=colors)


# %% codecell
# load  the image to recolor
hipr_seg_fn = hipr_out_dir + '/' + config_hipr['seg_fmt'].format(sample_name=sn)
hipr_seg = np.load(hipr_seg_fn)
# hipr_seg_resize_fmt = config['output_dir'] + '/' + config['hipr']['seg_resize']
# hipr_seg_resize = np.load(hipr_seg_resize_fmt.format(sample_name=sn))
# hipr_seg_resize.shape
# seg.shape

# %% codecell
# label to barcode dictionairy
dict_label_bc = dict(zip(props_reclass['label'], props_reclass['code_manual']))
# barcode to color dict
probe_design_dir = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR']
# df_bc_col = pd.read_csv(probe_design_dir + '/' + config_hipr['barcode_color_fn'])
bc_7b = [str(bc).zfill(7) for bc in df_bc_col['barcode_7b'].values]
# bc_5b = [str(bc).zfill(5) for bc in df_bc_col['barcode'].values]
# bc_7b = [bc[:2] + '00' + bc[2:] for bc in bc_5b]
dict_bc_col = dict(zip(bc_7b, df_bc_col['color']))
# label to color dict
dict_label_col = {l:dict_bc_col[bc] for l, bc in dict_label_bc.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)


# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/manual_classif_col'
ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)


# %% codecell
# plot taxon legend
bc_ordered = bc_counts.index.values
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
tax_ordered = [dict_bc_tax[int(bc)] for bc in bc_ordered]
col_ordered = [dict_bc_col[bc] for bc in bc_ordered]
fig, ax = ip.taxon_legend(
        taxon_names=tax_ordered,
        taxon_colors=col_ordered,
        taxon_counts=bc_counts
        )
out_bn = classif_out_dir + '/manual_classif_legend'
ip.save_png_pdf(out_bn)


# %% md

# check lautropia assignemnt

# %% codecell
# Subset segmentation
i = [400,0]
w = [100,100]
hipr_seg_sub = hipr_seg[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
sum_sub = sum_im[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
hsr_col_sub = hsr_col[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
# Check subset
plt.imshow(sum_sub)
plt.show()
plt.close()
plt.imshow(hsr_col_sub)
plt.show()
plt.close()
plt.imshow(hipr_seg_sub)

# %% codecell
# Get labels
labs = np.unique(hipr_seg_sub)
labs
# %% codecell
# Get manual assignment for labels
# dict_lab_clust = dict(zip(props_reclass['label'], labels_csf_kmeans))
dict_lab_clust = dict(zip(props_reclass['label'], props_reclass['code_manual']))
clusts = [dict_lab_clust[l] for l in labs[1:]]
clusts

# %% md
# =============================================================================
# ## Show the kmeans clustering
# =============================================================================

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_csf_kmeans))
# cluster to color dict
clusters = np.unique(labels_csf_kmeans, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
colors = colors[:len(clusters_sort)]
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/kmeans_clusters_col'
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot taxon legend
bc_ordered = bc_counts.index.values
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
tax_ordered = [dict_bc_tax[int(bc)] for bc in bc_ordered]
col_ordered = [dict_bc_col[bc] for bc in bc_ordered]
fig, ax = ip.taxon_legend(
        taxon_names=clusters_sort,
        taxon_colors=colors,
        taxon_counts=sorted(clusters[1], reverse=True)
        )
out_bn = classif_out_dir + '/kmeans_clusters_legend'
ip.save_png_pdf(out_bn)

# %% md

# Try different kmeans clustering

# %% codecell
n_bc = barcodes_all.shape[0]
cell_spec_full = props[avgint_cols].values
kmeans = KMeans(n_clusters=n_bc, random_state=42)
labels_spec_raw_kmeans = kmeans.fit_predict(cell_spec_full)
labels_spec_raw_kmeans.shape
# clust_assgn = pd.DataFrame({'label':props.label.values,'kmeans_cluster':labels_csf_kmeans})
# clust_assgn.to_csv(classif_out_dir + '/cell_cluster_assignment.csv')

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_spec_raw_kmeans))
# cluster to color dict
clusters = np.unique(labels_spec_raw_kmeans, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
colors = colors[:len(clusters_sort)]
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props['label'], props['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/kmeans_clusters_specraw_col'
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot taxon legend
bc_ordered = bc_counts.index.values
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
tax_ordered = [dict_bc_tax[int(bc)] for bc in bc_ordered]
col_ordered = [dict_bc_col[bc] for bc in bc_ordered]
fig, ax = ip.taxon_legend(
        taxon_names=clusters_sort,
        taxon_colors=colors,
        taxon_counts=sorted(clusters[1], reverse=True)
        )
out_bn = classif_out_dir + '/kmeans_clusters_specraw_legend'
ip.save_png_pdf(out_bn)


# %% codecell
# PLot the spectra from the cells as grouped by the kmeans clustering
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
classif_out_dir = hipr_out_dir + '/kmeans_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_spec_raw_kmeans)
for l in labs:
    bool = (labels_spec_raw_kmeans == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_kmeans_raw_cluster_' + str(l).zfill(2)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()



# %% md

# Try allowing more clusters

# %% codecell
n_bc = barcodes_all.shape[0]
cell_spec_full = props[avgint_cols].values
kmeans = KMeans(n_clusters=n_bc*2, random_state=42)
labels_spec_raw_kmeans_36 = kmeans.fit_predict(cell_spec_full)
print(n_bc*2)
labels_spec_raw_kmeans_36.shape

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_spec_raw_kmeans_36))
# cluster to color dict
clusters = np.unique(labels_spec_raw_kmeans_36, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
delta = 1/len(clusters_sort)
cmap = plt.get_cmap('gist_rainbow')
ci = [0,0.4,0.8,0.2,0.6,1,0.1,0.5,0.9,0.05,0.45,0.85,0.15,0.55,0.95,0.025, 0.425, 0.825, 0.125, 0.525, 0.925, 0.075, 0.475, 0.875, 0.175, 0.575, 0.975, 0.19, 0.59, 0.99, 0.09, 0.49,0.89, 0.11, 0.51, 0.91]
len(ci)
colors = [cmap(i) for i in ci]
# colors = [cmap(i*delta) for i in range(len(clusters_sort))]
# colors = list(plt.get_cmap('tab20').colors)
colors = colors[:len(clusters_sort)]
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=4)
hsr_col[(hsr_col[:,:,3] == 0), 3] = 1

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/kmeans_clusters_specraw_36_col'
ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot taxon legend
bc_ordered = bc_counts.index.values
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
tax_ordered = [dict_bc_tax[int(bc)] for bc in bc_ordered]
col_ordered = [dict_bc_col[bc] for bc in bc_ordered]
fig, ax = ip.taxon_legend(
        taxon_names=clusters_sort,
        taxon_colors=colors,
        taxon_counts=sorted(clusters[1], reverse=True)
        )
out_bn = classif_out_dir + '/kmeans_clusters_specraw_36_legend'
ip.save_png_pdf(out_bn)

# %% codecell
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
classif_out_dir = hipr_out_dir + '/kmeans_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_spec_raw_kmeans_more)
for l in labs:
    bool = (labels_spec_raw_kmeans_more == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_kmeans_raw_36_cluster_' + str(l).zfill(2)
    ip.save_png_pdf(out_bn)
    plt.show()


# %% md

# =============================================================================
# Plot spectra from simulations
# =============================================================================

# %% codecell
# get  simulation training data
ref_train_dir = config['hipr_dir'] + '/' + config_hipr['reference_training']['out_dir']
td_fn = ref_train_dir + '/PROBEDESIGN_welch2016_BITS_7_NSIMS_2000_OBJ_training_data.csv'
tdn_fn = ref_train_dir + '/PROBEDESIGN_welch2016_BITS_7_NSIMS_2000_OBJ_training_data_neg.csv'
training_data = pd.read_csv(td_fn).values
training_data_neg = pd.read_csv(tdn_fn).values
training_data_full = np.vstack([training_data, training_data_neg])
training_data_full.shape

# %% codecell
# Plot spectra
for bc in barcodes_all:
    bool = (training_data[:,-1] == int(bc))
    spec_ = training_data[bool,:-(1+4+6)]  # remove label, rough classifier, and 633 channel
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.2,'alpha':0.2,'color':'r'})
    ax.set_title(str(bc))
    ref_train_dir
    out_bn = ref_train_dir + '/spectra_training_data_barcode_' + str(bc).zfill(2)
    print(out_bn)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

training_data[:,-1][training_data[:,-1] == 100000]

(training_data[:,-1] == 100000).any()


# %% md

# =============================================================================
# Try hdbscan clustering
# =============================================================================

# %% codecell
classif_out_dir = hipr_out_dir + '/hdbscan_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)

# %% codecell
n_bc = barcodes_all.shape[0]
cell_spec_full = props[avgint_cols].values
cell_spec_full.shape

# %% codecell
# NOrmalize
csf_norm = cell_spec_full / np.max(cell_spec_full, axis=1)[:,None]
# scale
csf_scaler = StandardScaler().fit(csf_norm)
csf_scale = csf_scaler.transform(csf_norm)
# %% codecell
# umap reduction
umap_cell_full = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
# umap_cell_full = UMAP(n_neighbors=10, min_dist=0.1, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_cell_full = umap_cell_full.fit(csf_scale)
ut_cell_full = umap_transform_cell_full.embedding_
# %% codecell
# hdbscan
hdbscan_obj = HDBSCAN(min_cluster_size=5, min_samples=15)
labels_ut_hdbscan = hdbscan_obj.fit_predict(ut_cell_full)
labels_ut_hdbscan.shape

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell_full, labels_ut_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_ut_hdbscan))
# cluster to color dict
clusters = np.unique(labels_ut_hdbscan, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
# colors = list(plt.get_cmap('tab20').colors)
cmap = plt.get_cmap('gist_rainbow')
ci = [0,0.4,0.8,0.2,0.6,1,0.1,0.5,0.9,0.05,0.45,0.85,0.15,0.55,0.95,0.025, 0.425, 0.825, 0.125, 0.525, 0.925, 0.075, 0.475, 0.875, 0.175, 0.575, 0.975, 0.19, 0.59, 0.99, 0.09, 0.49,0.89, 0.11, 0.51, 0.91]
len(ci)
colors = [cmap(i) for i in ci]
colors = colors[:len(clusters_sort)]
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=4)
hsr_col[(hsr_col[:,:,3] == 0), 3] = 1

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/hdbscan_clusters_specraw_col'
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot taxon legend
bc_ordered = bc_counts.index.values
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
tax_ordered = [dict_bc_tax[int(bc)] for bc in bc_ordered]
col_ordered = [dict_bc_col[bc] for bc in bc_ordered]
fig, ax = ip.taxon_legend(
        taxon_names=clusters_sort,
        taxon_colors=colors,
        taxon_counts=sorted(clusters[1], reverse=True)
        )
out_bn = classif_out_dir + '/hdbscan_clusters_specraw_legend'
ip.save_png_pdf(out_bn)


# %% codecell
# PLot the spectra from the cells as grouped by the hdbscan clustering
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
labs = np.unique(labels_spec_raw_kmeans)
for l in labs:
    bool = (labels_spec_raw_kmeans == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_hdbscan_raw_cluster_' + str(l).zfill(2)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()




# %% md

# =============================================================================
# Try agglomerative clustering
# =============================================================================

# %% codecell
classif_out_dir = hipr_out_dir + '/agglomerative_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)

# # %% codecell
# # Get distance matrix
# shp = cell_spec_full.shape[0]
# csf_dmat_cos = np.zeros((shp,shp))
# for i in range(shp):
#     for j in range(shp):
#         si = cell_spec_full[i,:]
#         if not i == j:
#             sj = cell_spec_full[j,:]
#             csf_dmat_cos[i,j] = fhc.channel_cosine_intensity_5b_v2(si, sj)

# %% codecell
# alternate distance matrix using euclidean distance of cumulative spectrum
s = cell_spec_full[:4,:]
fig, ax = ip.general_plot()
fsi.plot_cell_spectra(ax,s[:2,:], {'lw':1,'alpha':1,'color':'r'})
print(fhc.euclid_dist_cumul_spec(s[0,:],s[1,:]))

# %% codecell
fig, ax = ip.general_plot()
fsi.plot_cell_spectra(ax,s[2:4], {'lw':1,'alpha':1,'color':'r'})
print(fhc.euclid_dist_cumul_spec(s[2,:],s[3,:]))

# %% codecell
s_norm = s / np.max(s, axis=1)[:,None]
fig, ax = ip.general_plot()
fsi.plot_cell_spectra(ax,s_norm[:2,:], {'lw':1,'alpha':1,'color':'r'})
plt.show()
plt.close
print(fhc.euclid_dist_cumul_spec(s_norm[0,:],s_norm[1,:]))
fig, ax = ip.general_plot()
fsi.plot_cell_spectra(ax,s_norm[2:4,:], {'lw':1,'alpha':1,'color':'r'})
plt.show()
plt.close
print(fhc.euclid_dist_cumul_spec(s_norm[2,:],s_norm[3,:]))

# %% codecell
n_las = 3
s_lnorm = np.hstack([s[:,c[i]:c[i+1]]/np.max(s[:,c[i]:c[i+1]], axis=1)[:,None] for i in range(n_las)])
fig, ax = ip.general_plot()
fsi.plot_cell_spectra(ax,s_lnorm[:2,:], {'lw':1,'alpha':1,'color':'r'})
plt.show()
plt.close
print(fhc.euclid_dist_cumul_spec(s_lnorm[0,:],s_lnorm[1,:]))
fig, ax = ip.general_plot()
fsi.plot_cell_spectra(ax,s_lnorm[2:4,:], {'lw':1,'alpha':1,'color':'r'})
plt.show()
plt.close
print(fhc.euclid_dist_cumul_spec(s_lnorm[2,:],s_lnorm[3,:]))

# %% codecell
shp = csf_scale.shape[0]
csf_dmat_ecsn = np.zeros((shp,shp))
for i in range(shp):
    for j in range(shp):
        si = csf_scale[i,:]
        if not i == j:
            sj = csf_scale[j,:]
            csf_dmat_ecsn[i,j] = fhc.euclid_dist_cumul_spec(si, sj)
plt.hist(np.ravel(csf_dmat_ecsn), bins=20)


# %% codecell
# Cluster
from sklearn.cluster import AgglomerativeClustering
distance_threshold = 6
aggl_obj = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        affinity='precomputed',
        linkage='average'
        )
labels_aggl_ecsn = aggl_obj.fit_predict(csf_dmat_ecsn)
np.unique(labels_aggl_ecsn).shape

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_aggl_ecsn))
# cluster to color dict
clusters = np.unique(labels_aggl_ecsn, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/agglomerative_clusters_specscale_col'
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot taxon legend
bc_ordered = bc_counts.index.values
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
tax_ordered = [dict_bc_tax[int(bc)] for bc in bc_ordered]
col_ordered = [dict_bc_col[bc] for bc in bc_ordered]
fig, ax = ip.taxon_legend(
        taxon_names=clusters_sort[:20],
        taxon_colors=colors[:20],
        taxon_counts=sorted(clusters[1], reverse=True)[:20]
        )
out_bn = classif_out_dir + '/agglomerative_clusters_specraw_legend'
ip.save_png_pdf(out_bn)


# %% codecell
# PLot the spectra from the cells as grouped by the kmeans clustering
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
classif_out_dir = hipr_out_dir + '/agglomerative_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_aggl_ecs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (labels_aggl_ecs == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_agglomerative_raw_cluster_' + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# umap reduction
umap_obj = UMAP(n_neighbors=10, metric=fhc.euclid_dist_cumul_spec)
umap_transform_csfn_ecs = umap_obj.fit(csf_scale)
ut_csfn_ecs = umap_transform_csfn_ecs.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_csfn_ecs, labels_aggl_ecsn, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)




# %% md

# try with the raw spectra non normalized


# %% codecell
shp = cell_spec_full.shape[0]
csf_dmat_ecs = np.zeros((shp,shp))
for i in range(shp):
    for j in range(shp):
        si = cell_spec_full[i,:]
        if not i == j:
            sj = cell_spec_full[j,:]
            csf_dmat_ecs[i,j] = fhc.euclid_dist_cumul_spec(si, sj)
plt.hist(np.ravel(csf_dmat_ecs), bins=20)


# %% codecell
# Cluster
from sklearn.cluster import AgglomerativeClustering
distance_threshold = 0.1
aggl_obj = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        affinity='precomputed',
        linkage='average'
        )
labels_aggl_ecs = aggl_obj.fit_predict(csf_dmat_ecs)
np.unique(labels_aggl_ecs).shape

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_aggl_ecs))
# cluster to color dict
clusters = np.unique(labels_aggl_ecs, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props['label'], props['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/agglomerative_clusters_specraw_col'
ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot taxon legend
fig, ax = ip.taxon_legend(
        taxon_names=clusters_sort[:20],
        taxon_colors=colors[:20],
        taxon_counts=sorted(clusters[1], reverse=True)[:20]
        )
out_bn = classif_out_dir + '/agglomerative_clusters_specraw_legend'
ip.save_png_pdf(out_bn)


# %% codecell
# PLot the spectra from the cells as grouped by the kmeans clustering
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
classif_out_dir = hipr_out_dir + '/agglomerative_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
# labs = np.unique(labels_aggl_ecs, return_counts=True)
# labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in clusters_sort[:20]:
    bool = (labels_aggl_ecs == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.25,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_agglomerative_raw_cluster_' + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# umap reduction
umap_obj = UMAP(n_neighbors=10, metric=fhc.euclid_dist_cumul_spec)
umap_transform_csf_ecs = umap_obj.fit(cell_spec_full)
ut_csf_ecs = umap_transform_csf_ecs.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_csf_ecs, labels_aggl_ecs, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)


# %% codecell
# Get averaged spectra for cluster
spec_aggl = {}
for l in clusters_sort:
    bool = (labels_aggl_ecs == l)
    spec_ = cell_spec_full[bool,:]
    spec_avg = np.mean(spec_, axis=0)
    spec_aggl[l] = spec_avg

# %% codecell
# plot av spec for cluster
for l in clusters_sort[:20]:
    fig, ax = ip.general_plot(dims=(10,5))
    spec_ = spec_aggl[l][None,:]
    print(spec_.shape)
    fsi.plot_cell_spectra(ax, spec_, {'lw':1,'alpha':1,'color':'r'})
    plt.show()
    plt.close()


# %% md

# =============================================================================
# Try to use nearest neighbor classifier on the training dataset
# =============================================================================

# %% codecell
from sklearn.neighbors import NearestNeighbors

n_chan = 4
nc = 1 + n_chan + 6
tdf_codes = training_data_full[:,-1]
tdf_arr = training_data_full[training_data_full[:,-1] != 1100111]
tdf_arr = tdf_arr[:,:-nc]


# %% codecell
neigh_obj = NearestNeighbors(n_neighbors=1, metric=fhc.euclid_dist_cumul_spec)
neigh_fit = neigh_obj.fit(tdf_arr)
dist, ind = neigh_fit.kneighbors(csf_norm)

# %% codecell
classifs = tdf_codes[ind]

# %% codecell
# label to cluster  dictionairy
classifs_float = [float(re.sub('_error','.404', str(bc[0]))) for bc in classifs]
dict_label_clust = dict(zip(props['label'], classifs_float))
# cluster to color dict
clusters = np.unique(classifs_float, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props['label'], props['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
classif_out_dir = hipr_out_dir + '/nn_cluster_ecs'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/{}_nn_classif_ecs_col'.format(sn)
ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot taxon legend
taxon_names = [dict_bc_tax[int(bc)] for bc in clusters_sort[:20]]
fig, ax = ip.taxon_legend(
        taxon_names=taxon_names,
        taxon_colors=colors[:20],
        taxon_counts=sorted(clusters[1], reverse=True)[:20]
        )
out_bn = classif_out_dir + '/{}_nn_classif_ecs_legend'.format(sn)
ip.save_png_pdf(out_bn)

# %% codecell
# PLot the spectra from the cells as classified
# hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
# classif_out_dir = hipr_out_dir + '/agglomerative_cell_spectra'
# if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
# labs = np.unique(labels_aggl_ecs, return_counts=True)
# labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in clusters_sort[:20]:
    bool = (classifs_float == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.25,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/{}_nn_classif_ecs_spectra_bc_{}'.format(sn,int(l))
    print(out_bn)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()


# %% md

# =============================================================================
# Try hdbscan clustering on the new metric
# =============================================================================

# normalized spetra ecs metric

# %% codecell
# Cluster
hdbscan_obj = HDBSCAN(min_cluster_size=5, min_samples=10, metric=fhc.euclid_dist_cumul_spec)
labels_ecsn_hdbscan = hdbscan_obj.fit_predict(csf_scale)

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_ecsn_hdbscan))
# cluster to color dict
clusters = np.unique(labels_ecsn_hdbscan, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/hdbscan_ecsn_clusters_specraw_col'
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# umap reduction
umap_obj = UMAP(n_neighbors=10, metric=fhc.euclid_dist_cumul_spec)
umap_transform_csfnorm_ecs = umap_obj.fit(csf_scale)
ut_csfn_ecs = umap_transform_csfnorm_ecs.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_csfn_ecs, labels_ecsn_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)


# %% md

# raw spectra ecs metric

# %% codecell
# Cluster
hdbscan_obj = HDBSCAN(min_cluster_size=20, min_samples=10, metric=fhc.euclid_dist_cumul_spec)
labels_ecs_hdbscan = hdbscan_obj.fit_predict(cell_spec_full)

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_ecs_hdbscan))
# cluster to color dict
clusters = np.unique(labels_ecsn_hdbscan, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/hdbscan_ecs_clusters_specraw_col'
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# umap reduction
umap_obj = UMAP(n_neighbors=10, metric=fhc.euclid_dist_cumul_spec)
umap_transform_csf_ecs = umap_obj.fit(cell_spec_full)
ut_csf_ecs = umap_transform_csf_ecs.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_csf_ecs, labels_ecs_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)


# %% md

# scaled spectra cos metric

# %% codecell
# Cluster
hdbscan_obj = HDBSCAN(min_cluster_size=20, min_samples=10, metric=fhc.channel_cosine_intensity_5b_v2)
labels_cosn_hdbscan = hdbscan_obj.fit_predict(csf_scale)

# %% codecell
# label to cluster  dictionairy
dict_label_clust = dict(zip(props['label'], labels_cosn_hdbscan))
# cluster to color dict
clusters = np.unique(labels_cosn_hdbscan, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab20').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props_reclass['label'], props_reclass['bbox']))
# re color image
hsr_col = fsi.recolor_image(hipr_seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(hsr_col, scalebar_resolution=config_hipr['resolution'])
out_bn = classif_out_dir + '/hdbscan_ecs_clusters_specraw_col'
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# umap reduction
umap_obj = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_csfn_cos = umap_obj.fit(csf_scale)
ut_csf_cosn = umap_transform_csfn_cos.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_csf_cosn, labels_cosn_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)









# %% codecell
# smooth the pixel spectra
sigma=1.4
reg_smooth = np.empty(reg.shape)
for i in range(reg.shape[2]):
    reg_smooth[:,:,i] = gaussian_filter(reg[:,:,i], sigma=sigma)
# %% codecell
# Get the pixel by pixel spectra for the cell seg
mask = seg > 0
spec_pix_full = reg_smooth[mask, :]
spf_norm = spec_pix_full / np.max(spec_pix_full, axis=1)[:,None]
spf_scaler = StandardScaler().fit(spf_norm)
spf_scale = spf_scaler.transform(spf_norm)
labels_pix_full = [0] * spf_scale.shape[0]
spf_scale.shape

# %% codecell
# umap reduction
umap_pix_full = UMAP(n_neighbors=10, metric=fhc.channel_cosine_intensity_5b_v2)
umap_transform_pix_full = umap_cell_full.fit(spf_scale)
ut_pix_full = umap_transform_pix_full.embedding_
# labels_ut_cell = np.repeat(0,cell_spec.shape[0]).tolist()

# %% codecell
# plot with labels
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_pix_full, labels_pix_full, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)

# %% codecell
# cluster
labels_utcsf_hdbscan = HDBSCAN(min_cluster_size=1000, min_samples=2000).fit_predict(ut_cell_full)
# labels_csf_hdbscan = HDBSCAN(min_cluster_size=10, min_samples=10).fit_predict(cell_spec_full)

# %% codecell
# Plot the umap
dims=(10,10)
ms=20
ft=14
alpha=0.8
xlims=[]
ylims=[]
fig, ax, col_df = fhc.plot_umap(ut_cell_full, labels_utcsf_hdbscan, dims=dims,
        markersize=ms, xlims=xlims, ylims=ylims, ft=ft, alpha=alpha)



# %% md

# Quantify fraction of e coli with spots

# %% md

# Get neighbors for spots

# %% codecell
# Load shift vectors





# %% md
