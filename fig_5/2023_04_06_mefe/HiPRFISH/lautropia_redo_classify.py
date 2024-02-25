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
import joblib
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict
import numba as nb

# %% md

# Move to the workdir

# %% codecell
# Absolute path
cluster_dir = ''
project_workdir = cluster_dir + '/workdir/bmg224/manuscripts/mgefish/code/fig_5/2023_04_06_mefe/HiPRFISH'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory


# %% md

# Load all the variables from the segmentation pipeline

# %% codecell
config_fn = '../config_hiprmeta.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

config_hipr_fn = config['config_hipr'] # relative path to config file from workdir
with open(config_hipr_fn, 'r') as f:
    config_hipr = yaml.safe_load(f)

config_mega_fn = config['config_mega'] # relative path to config file from workdir
with open(config_mega_fn, 'r') as f:
    config_mega = yaml.safe_load(f)

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%load_ext autoreload
%autoreload 2

sys.path.append(config['pipeline_path'] + '/' + config_mega['functions_path'])
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
# Pick an image
mega_out_dir = config['mega_dir'] + '/' + config_mega['output_dir']
input_table = pd.read_csv(config['mega_dir'] + '/' + config_mega['input_table_fn'])
input_table

# %% codecell
sn_i = 0
sn = input_table['sample_name'][sn_i]
sn

# %% md

# ==============================================================================
# ## Fix image registration
# ==============================================================================

# %% codecell
# raw images
hipr_out_dir = config['hipr_dir'] + '/' + config_hipr['output_dir']
data_dir = (hipr_out_dir + '/' + config_hipr['__default__']['DATA_DIR'])
raw_regex = data_dir + '/' + sn + config_hipr['laser_regex']
# out_dir = (config['output_dir'] + '/' + config['prep']['out_dir'])
# raw_regex = os.path.splitext(out_dir + '/' + sn + config['laser_regex'])[0] + '.npy'
# raws = [np.load(fn) for fn in raw_fns]
raw_fns = sorted(glob.glob(raw_regex))
print(raw_fns)


# %% codecell
# Load with bioformats
import javabridge
import bioformats
import aicspylibczi as aplc
javabridge.start_vm(class_path=bioformats.JARS)
raws = [bioformats.load_image(fn) for fn in raw_fns]
# raws = [aplc.CziFile(fn) for fn in raw_fns]
# print([r.get_dims_shape() for r in raws])
# print([r.is_mosaic() for r in raws])
# tile_bboxes = [r.get_all_tile_bounding_boxes() for r in raws]
# dir(tile_bboxes[0])

# [[k.dimension_coordinates, v.x, v.y, v.w, v.h] for k,v in tile_bboxes[0].items()]
# %% codecell
print(raws[0].shape)

# %% codecell
# show rgb overlay
ulims = (0.2,0.25,0.25)

im_inches=20
raws_max = [np.sum(im, axis=2) for im in raws]
raws_max_norm = [im / np.max(im) for im in raws_max]
raws_max_norm_adj = []
for im, ul in zip(raws_max_norm, ulims):
    im[im>ul] = ul
    raws_max_norm_adj.append(im/ul)

gammas = [1.5,1.5,1.5]  # Gamma correction
raws_max_norm_adj = [r ** (1/g) for r, g in zip(raws_max_norm_adj, gammas)]
fig, ax = ip.general_plot()
ax.plot([0,1],[0,1])
x = np.linspace(0,1,100)
for g in gammas:
    ax.plot(x, x**(1/g))
# ax.set_aspect('equal')


rgb = np.dstack(raws_max_norm_adj)
ip.plot_image(rgb, im_inches=im_inches)
output_dir = config['output_dir'] + '/test_classif/mat_mul'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# out_bn = output_dir + '/' + sn + '_rgb'
# ip.save_png_pdf(out_bn)
# plt.show()
# plt.close()



# %% codecell
# Register images
shift_vectors = fsi._get_shift_vectors(raws_max)
print(shift_vectors)
# %% codecell
raws_shift = fsi._shift_images(raws, shift_vectors, max_shift=500)
print(raws_shift[0].shape)

# %% codecell
im_inches = 10
# show unshifted rgb overlay on zoom
c = [500, 500]  # corner
# c = [1400, 1400]  # corner
w = [500,500]  # height and width
# ulims = (1,1,1)


stack_zoom = np.dstack(raws_max_norm_adj)[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
ip.plot_image(stack_zoom, im_inches=im_inches)
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape//(im_inches*2))
plt.show()
plt.close()


# %% codecell
# shifted
raws_shift_max = [np.sum(im, axis=2) for im in raws_shift]
raws_shift_max_norm = [im / np.max(im) for im in raws_shift_max]
raws_shift_max_norm_adj = []
for im, ul in zip(raws_shift_max_norm, ulims):
    im[im>ul] = ul
    raws_shift_max_norm_adj.append(im/ul)

rgb = np.dstack(raws_shift_max_norm_adj)
raws_shift_max_norm_adj_sub = rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
ip.plot_image(raws_shift_max_norm_adj_sub, im_inches=im_inches)
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape//(im_inches*2))
plt.show()
plt.close()

# %% codecell
# save rgb overlay

im_inches=5
ip.plot_image(rgb, im_inches=im_inches)
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_rgb'
# ip.save_png_pdf(out_bn)
plt.show()
plt.close()


# %% codecell
stack = np.dstack(raws_shift)
stack_zoom = stack[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
stack.shape


# %% md

# ==============================================================================
# ## Overlay with MeGAFISH
# ==============================================================================

### Get Shift vectors for mega images

# %% codecell
# Get the Rescaled hiprfish image
from cv2 import resize, INTER_CUBIC, INTER_NEAREST

config_hipr_fn = 'config_hipr.yaml'
with open(config_hipr_fn, 'r') as f:
    config_hipr = yaml.safe_load(f)

mega_dir = '../MeTAFISH'
config_mega_fn = mega_dir + '/config_meta.yaml'
with open(config_mega_fn, 'r') as f:
    config_mega = yaml.safe_load(f)

hiprmega_dir = '..'
config_hiprmega_fn = hiprmega_dir + '/config_hiprmeta.yaml'
with open(config_hiprmega_fn, 'r') as f:
    config_hiprmega = yaml.safe_load(f)
config=config_hiprmega


def center_image(im, dims, ul_corner):
    shp = im.shape
    if not all([dims[i] == shp[i] for i in range(len(dims))]):
        shp_new = dims if len(shp) == 2 else dims + (shp[2],)
        temp = np.zeros(shp_new)
        br_corner = np.array(ul_corner) + np.array(shp[:2])
        temp[ul_corner[0]:br_corner[0], ul_corner[1]:br_corner[1]] = im
        im = temp
    return im

def resize_hipr(im, hipr_res, mega_res, dims='none', out_fn=False, ul_corner=(0,0)):
    # im = np.load(in_fn)
    factor_resize = hipr_res / mega_res
    hipr_resize = resize(
            im,
            None,
            fx = factor_resize,
            fy = factor_resize,
            interpolation = INTER_NEAREST
            )
    if isinstance(dims, str): dims = hipr_resize.shape
    hipr_resize = center_image(hipr_resize, dims, ul_corner)
    # if out_fn: np.save(out_fn, hipr_resize)
    return hipr_resize

mega_res = config_mega['resolution']
hipr_res = config_hipr['resolution']
hipr_max_resize = resize_hipr(
        np.max(stack, axis=2), hipr_res, mega_res
        )

# %% codecell
# Get mega cell image
# config_hipr = config

sn_old = '2023_04_06_mefe_pool_full_slide_1_fov_07'

mega_out_dir = mega_dir + '/' + config_mega['output_dir']

mega_raw_fn = mega_out_dir + '/' + config_mega['raw_fmt'].format(sample_name=sn_old)
mega_raw = np.load(mega_raw_fn)
reg_ch = config['mega_reg_ch_index']
mega_cell = mega_raw[:,:,config_mega['cell_seg']['channels'][reg_ch]]

# %% codecell
# Which is the smaller image?
mshp = mega_cell.shape
hshp = hipr_max_resize.shape
im_list = [mega_cell, hipr_max_resize]
i_sml = np.argmin([mshp[0],hshp[0]])
i_lrg = np.argmax([mshp[0],hshp[0]])
sml = im_list[i_sml]
lrg = im_list[i_lrg]
# Get half the difference between sizes
shp_dff = np.abs(np.array(hshp) - np.array(mshp)) // 2
# Shift the smaller image so that it sits at the center of the larger image
sml_shift = np.zeros(lrg.shape)
corn_ind = np.array(shp_dff) + np.array(sml.shape)
sml_shift[shp_dff[0]:corn_ind[0], shp_dff[1]:corn_ind[1]] = sml
# reassign mega and hipr image var names
im_shift_list = [0,0]
im_shift_list[i_sml] = sml_shift
im_shift_list[i_lrg] = lrg
mega_shift = im_shift_list[0]
hipr_shift = im_shift_list[1]
# Get the shift vectors for the mega image
image_list = [hipr_shift, mega_shift]
shift_vectors = fsi._get_shift_vectors(image_list)
print(shift_vectors)

# %% md

### Shift the mega images

# %% codecell
# Load megafish spot seg
s_ch = 1
c_ch = 0
# mega_cs_fn = mega_out_dir + '/' + config_mega['cell_seg_area_filt_fmt'].format(sample_name=sn,cell_chan=c_ch)
mega_ss_fn = mega_out_dir + '/' + config_mega['spot_mask_bg_ecc_fmt'].format(sample_name=sn_old, spot_chan=s_ch)
# mega_cs = np.load(mega_cs_fn)
mega_ss = np.load(mega_ss_fn)

# %% codecell
# globally define vars for the shift function
def shift_mega(im):
    '''Globally define:  mega_shift_vector, max_shift, dims, ul_corner'''
    # im = np.load(in_fn)
    if len(im.shape) == 2: im = im[...,None]
    im = center_image(im, dims, ul_corner)
    return fsi._shift_images([im], mega_shift_vector, max_shift=max_shift)

def get_mega_props(seg, raw_shift, ch_):
    print(raw_shift.shape)
    raw = raw_shift[:,:,ch_]
    seg = seg.astype(np.int64)[:,:,0]
    return sf.measure_regionprops(seg, raw=raw)


max_shift = config['max_shift']
mega_shift_vector = [shift_vectors[1]]
dims = lrg.shape
ul_corner = shp_dff
# run the shift function
raw_shift = shift_mega(mega_raw)
# cs_shifts = shift_mega(mega_cs)
ss_shifts = shift_mega(mega_ss)

# %% codecell
# Run the props function
# cs_shifts[0].shape
# css_props = get_mega_props(cs_shifts[0], raw_shift[0], ch_=c_ch)
sss_props = get_mega_props(ss_shifts[0], raw_shift[0], ch_=s_ch)

# %% codecell
# resize hiprfish images
# hipr_plot_intensities_resize = resize_hipr(
#         plot_intensities, hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
#         )
hipr_maz_resize = resize_hipr(
        np.max(stack, axis=2), hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
        )
hipr_rgb_resize = resize_hipr(
        rgb, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )

hipr_stack_resize = resize_hipr(
        stack, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )

hipr_stacksmooth_resize = resize_hipr(
        stack_smooth, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )

# %% codecell
# plot cell overlay
im_inches=10

m_ul=1
mega_raw_cell_norm = raw_shift[0][:,:,0].copy()
mega_raw_cell_norm /= np.max(mega_raw_cell_norm)
mega_raw_cell_norm[mega_raw_cell_norm > m_ul] = m_ul
mega_raw_cell_norm /= m_ul

h_ul=1
hipr_raw_cell_norm = hipr_maz_resize.copy()
hipr_raw_cell_norm /= np.max(hipr_raw_cell_norm)
hipr_raw_cell_norm[hipr_raw_cell_norm > m_ul] = m_ul
hipr_raw_cell_norm /= m_ul

overl_stack = np.zeros(hipr_maz_resize.shape + (3,))
overl_stack[:,:,0] = hipr_raw_cell_norm
overl_stack[:,:,1] = mega_raw_cell_norm
fig, ax, _ = ip.plot_image(overl_stack, im_inches=im_inches)

# %% codecell
# Plot spot overlay on rgb
im_inches=20
ceil=1
size_min=0.1
marker_size=100
marker='.'
spot_col=(1,0,1)
edge_col='none'
linewidths=1.5

spot_props_shift = sss_props
fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
ref_pts = [c for c in spot_props_shift['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
ref_pts_arr = np.array(ref_pts)
# # Plot size as spot intensity
# spot_int = spot_props_shift.max_intensity.values.copy()
# spot_int -= np.min(spot_int)
# spot_int /= np.max(spot_int)
# spot_int[spot_int > ceil] = ceil
# spot_int /= ceil
# spot_int[spot_int < size_min] = size_min
# marker_size_arr = marker_size * spot_int
marker_size_arr=marker_size
ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
            marker=marker, s=marker_size_arr, color=spot_col,
            linewidths=linewidths, edgecolors=edge_col
            )
# out_dir = config['output_dir'] + '/pixel_classif_test'



# %% md

# ==============================================================================
# ## Matrix mult pixel classifier
# ==============================================================================

# %% codecell
# smoothing
sigma=3

# stack=stack_zoom

stack_smooth = np.empty(stack.shape)
for i in range(stack.shape[2]):
    stack_smooth[:,:,i] = gaussian_filter(stack[:,:,i], sigma=sigma)


# %% codecell
# Get maximum intensity mask
max_thresh = 0.03
im_inches=10

max_unsmooth = np.sum(stack, axis=2)
sum_sub = np.sum(stack, axis=2)
max_sub = np.max(stack_smooth, axis=2)
y = np.sort(np.ravel(max_sub))
x = np.arange(y.shape[0])
fig, ax = ip.general_plot()
ax.plot(x,y)
xlims = ax.get_xlim()
ax.plot(xlims, [max_thresh]*2)
plt.show()
plt.close()

mask = max_sub > max_thresh
max_masked = max_sub * mask
ip.subplot_square_images([max_sub, mask, max_masked], (1,3), clims=[[0,0.5],[],[0,0.5]], im_inches=im_inches)

# %% codecell
spec_pix = stack_smooth[mask > 0]
spec_pix_norm = spec_pix / np.max(spec_pix, axis=1)[:,None]

# %% codecell
from sklearn.preprocessing import StandardScaler

pix_scaler = StandardScaler().fit(spec_pix)
spec_pix_scaled = pix_scaler.transform(spec_pix)


# %% codecell
# get reference spectra
ref_dir = config['hipr_dir'] + '/' + config_hipr['hipr_ref_dir']
fmt = config_hipr['ref_files_fmt']
probe_design_dir = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config_hipr['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
barcodes = probe_design['code'].unique()
barcodes_str = [str(bc).zfill(7) for bc in barcodes]
barcodes_10bit = [bc[0] + '0' + bc[1:4] + '00' + bc[4:] for bc in barcodes_str]
barcodes_b10 = [int(str(bc),2) for bc in barcodes_10bit]
sci_names = [probe_design.loc[probe_design['code'] == bc,'sci_name'].unique()[0]
        for bc in barcodes]
st = config_hipr['ref_chan_start'] + config_hipr['chan_start']
en = config_hipr['ref_chan_start'] + config_hipr['chan_end']
ref_avgint_cols = [i for i in range(st,en)]

ref_spec = []
for bc in barcodes_b10:
    fn = ref_dir + '/'+ fmt.format(bc)
    ref = pd.read_csv(fn, header=None)
    ref = ref[ref_avgint_cols].values
    ref_spec.append(ref)
    # ref_norm = ref / np.max(ref, axis=1)[:,None]
    # weights.append(np.mean(ref_norm, axis=0))

# # %% codecell
# # max normalized reference
ref_norm = [r / np.max(r, axis=1)[:,None] for r in ref_spec]
weights_max_norm = [np.mean(r, axis=0) for r in ref_norm]

for w, n in zip(weights_max_norm, sci_names):
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
    ax.set_title(n)

# %% codecell
# sum normalized
ref_sum_norm = []
for r in ref_spec:
    r_ = r - np.min(r, axis=1)[:,None]
    ref_sum_norm.append(r_ / np.sum(r_, axis=1)[:,None])
weights_sum_norm = [np.mean(r, axis=0) for r in ref_sum_norm]
len(weights_sum_norm)

for w, n in zip(weights_sum_norm, sci_names):
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
    ax.set_title(n)

# %% codecell
# Only maximum values used in evaluation
top_n = 1
weights_top_n = []
for w in weights_max_norm:
    l = len(w)
    ind = np.argpartition(w, l-top_n)[:l-top_n]
    w_ = w.copy()
    w_[ind] = 0
    weights_top_n.append(w_)

for w, n in zip(weights_top_n, sci_names):
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
    ax.set_title(n)

# %% codecell
# Classify each pixel by picking the maximum weight
weights_pick = weights_sum_norm
input = spec_pix_scaled

weights_t = np.array(weights_pick).T
classif_mat = np.matmul(input, weights_t)
classifs_index = np.argmax(classif_mat, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell
# filter pixels that are not different enough from the mean
thr_scale = 1  # maximum must be this

pix_scaled_max = np.max(spec_pix_scaled, axis=1)
classifs[pix_scaled_max < thr_scale] = 'unclassified'

# %% codecell
n = 1000
labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    # spec_ = spec_pix_scaled[bool,:]
    # spec_ = spec_pix_norm[bool,:]
    spec_ = spec_pix[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.5,'color':'r'})
    ax.set_title(l)
    # output_dir = config['output_dir'] + '/test_classif/mat_mul'
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # out_bn = output_dir + '/' + sn + '_' + l + '_spectra'
    # ip.save_png_pdf(out_bn)
    #

# %% codecell
# Re project the pixels onto the image
# Colors
# barcode_7bit = [bc[:2] + bc[4:] for bc in barcodes_str]
# barcode_color_fn = probe_design_dir + '/' + config_hipr['barcode_color_fn']
# barcode_color = pd.read_csv(barcode_color_fn)
# col_dict = {}
# for bc, scn in zip(barcode_7bit, sci_names):
#     col_dict[scn] = eval(barcode_color.loc[barcode_color['barcode'] == int(bc),
#             'color'].values[0])
# Reduce the white level
ul = 0.25
plot_intensities = np.sum(stack, axis=2)
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
tab20 = plt.get_cmap('tab20').colors
tab_sort = [tab20[i] for i in np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()]
grey = tab_sort.pop(7)
tab_sort = [grey] + tab_sort
col_dict = {l:c for l, c in zip(labs_sort, tab_sort)}
im_classif_rgb = np.zeros(max_sub.shape + (len(tab20[0]),))
# im_classif_rgb = np.zeros(max_sub.shape + (len(eval(barcode_color.color.values[0])),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_classif_rgb[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell
im_inches=10
ip.plot_image(im_classif_rgb, im_inches=im_inches)
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_classif_col'
# ip.save_png_pdf(out_bn)

# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_classif_col_legend'
# ip.save_png_pdf(out_bn)





# %% md

# ==============================================================================
# ## Overlay classif with MeGAFISH
# ==============================================================================

### Get Shift vectors for mega images


# %% codecell
# resize hiprfish images
dims = lrg.shape
ul_corner = shp_dff
hipr_classif_resize = resize_hipr(
        im_classif_rgb, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )
hipr_maz_resize = resize_hipr(
        np.max(stack, axis=2), hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )


# %% codecell
# Plot spot overlay on classif
im_inches=20
ceil=1
size_min=0.1
marker_size=100
marker='.'
spot_col=(1,0,1)
edge_col='none'
linewidths=1.5

spot_props_shift = sss_props
fig, ax, cbar = ip.plot_image(
        hipr_classif_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
ref_pts = [c for c in spot_props_shift['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
ref_pts_arr = np.array(ref_pts)
# # Plot size as spot intensity
# spot_int = spot_props_shift.max_intensity.values.copy()
# spot_int -= np.min(spot_int)
# spot_int /= np.max(spot_int)
# spot_int[spot_int > ceil] = ceil
# spot_int /= ceil
# spot_int[spot_int < size_min] = size_min
# marker_size_arr = marker_size * spot_int
marker_size_arr=marker_size
ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
            marker=marker, s=marker_size_arr, color=spot_col,
            linewidths=linewidths, edgecolors=edge_col
            )
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spot_overlay_dot'
# ip.save_png_pdf(out_bn)


# %% codecell
# show raw spot on top of classif
ul = 0.15
ll = 0.1
spot_raw = raw_shift[0][:,:,1].copy()
spot_raw -= np.min(spot_raw)
spot_raw /= np.max(spot_raw)
spot_raw[spot_raw > ul] = ul
spot_raw[spot_raw < ll] = 0
spot_raw /= ul
spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
spot_raw_overlay[:,:,0] = spot_raw
spot_raw_overlay[:,:,2] = spot_raw
spot_raw_overlay[:,:,3] = spot_raw
fig, ax, cbar = ip.plot_image(
        hipr_classif_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
ax.imshow(spot_raw_overlay)
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay'
# ip.save_png_pdf(out_bn)

# %% codecell
# show raw spot on top of classif zoom
ul = 0.15
ll = 0.1
c=[1000,1000]
d=[1000,1000]
zoom_coords=[c[0], c[0]+d[0], c[1], c[1]+d[1]]
spot_raw = raw_shift[0][:,:,1].copy()
spot_raw -= np.min(spot_raw)
spot_raw /= np.max(spot_raw)
spot_raw[spot_raw > ul] = ul
spot_raw[spot_raw < ll] = 0
spot_raw /= ul
spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
spot_raw_overlay[:,:,0] = spot_raw
spot_raw_overlay[:,:,2] = spot_raw
spot_raw_overlay[:,:,3] = spot_raw
fig, ax, cbar = ip.plot_image(
        hipr_classif_resize, scalebar_resolution=mega_res, im_inches=im_inches,
        zoom_coords=zoom_coords
        )
ax.imshow(spot_raw_overlay)
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay'
# ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Spatial association between spots and pixels
# ==============================================================================

# Calculate actual spatial assoc

# %% codecell
# Get nearest neighbor classifier for pixels
r_um = 2

spot_coords = ref_pts_arr
hipr_mask_resize = resize_hipr(
        mask*1, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )
pix_coords = np.argwhere(hipr_mask_resize)

r_pix = int(r_um / mega_res)

# nbrs = NearestNeighbors(radius=r_pix).fit(pix_coords)
# # Search in radius around each spot
# inds = nbrs.radius_neighbors(spot_coords, return_distance=False)
# inds



# %% codecell
# Get resized classifs list for pixel inds
barcodes_int = [int(bc) for bc in barcodes]
dict_sn_bc = dict(zip(sci_names, barcodes_int))
dict_sn_bc['unclassified'] = 2
im_bc = np.zeros(max_sub.shape)
classifs_bc = [dict_sn_bc[sn] for sn in classifs]
for bc, i in zip(classifs_bc, pix_ind):
    x, y = i
    # col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_bc[x,y] = bc
hipr_bc_resize = resize_hipr(
        im_bc, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )

# %% codecell
# Get circle of pixels around spots
int_array = nb.types.int64[:]
@nb.njit()
def get_assocs(spot_coords, hipr_bc_resize, r_pix=50):
    # assocs_dict = defaultdict(lambda:0)
    assocs_dict = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
    for x, y in spot_coords:
        for i in range(2*r_pix):
            for j in range(2*r_pix):
                x_i = int(x - r_pix + i)
                y_j = int(y - r_pix + j)
                if ((x-x_i)**2 + (y-y_j)**2)**(1/2) <= r_pix:
                    # bc = hipr_bc_resize[y_j, x_i]
                    bc = np.int64(hipr_bc_resize[y_j, x_i])
                    # print(bc)
                    try:
                        assocs_dict[bc] += 1
                    except:
                        assocs_dict[bc] = 1
    return assocs_dict

# %% codecell
true_assoc_dict = get_assocs(spot_coords, hipr_bc_resize, r_pix=r_pix)
for bc in barcodes:
    try:
        _ = true_assoc_dict[bc]
    except:
        true_assoc_dict[bc] = 0
# values, counts = np.unique(assocs, return_counts=True)
# true_dict = dict(zip(values,counts))
true_assoc_dict

# %% codecell
# # Sum spatial assoc counts for each taxon
# values, counts = np.unique(classifs[inds], return_counts=True)
# true_dict = dict(zip(values,counts))

# %% md

# Simulate random distribution of spots

# %% codecell
# run knn on pixels
# inds_pix = nbrs.radius_neighbors(pix_coords, return_distance=False)
# iterate n times
n=1000
sim_dict = defaultdict(list)
for i in tqdm(range(n)):
    # Randomly select pixels
    i_sim = np.random.randint(0,pix_coords.shape[0], size=spot_coords.shape[0])
    # Generate counts for random selection
    sim_spot_coords = pix_coords[i_sim]
    sim_assoc_dict = get_assocs(sim_spot_coords, hipr_bc_resize, r_pix=r_pix)
    for bc in barcodes:
        try:
            _ = sim_assoc_dict[bc]
        except:
            sim_assoc_dict[bc] = 0
    for k, v in sim_assoc_dict.items():
        sim_dict[k].append(v)

# %% md

# Plot

# %% codecell
# Plot simulated values for random
# alpha=1
# true_col='k'
# true_lw=2
# lw=1
# dims=(3,2)
# ft=7
# nbins=100
# n_pix = np.sum([v for k, v in true_assoc_dict.items()])
# dict_bc_sciname = dict(zip(barcodes_int,sci_names))
# for bc_tax in barcodes_int:
#     sci_name = dict_bc_sciname[bc_tax]
#     color = col_dict[sci_name]
#     rand_counts = sim_dict[bc_tax]
#     # Get fraction of total spots
#     rand_frac = np.array(rand_counts) / n_pix
#     # plot distribution of spot assignment
#     nbins = np.unique(rand_frac).shape[0] // 4
#     bins = np.linspace(np.min(rand_frac), np.max(rand_frac), nbins)
#     # bins = np.arange(np.min(rand_frac), np.max(rand_frac))
#     # hist, bin_edges = np.histogram(rand_frac)
#     hist, bin_edges = np.histogram(rand_frac, bins=bins)
#     x_vals = ip.get_line_histogram_x_vals(bin_edges)
#     fig, ax = ip.general_plot(dims=dims, ft=ft, lw=lw)
#     ax.plot(x_vals, hist, color=color)
#     ax.fill_between(
#             x_vals,
#             np.zeros((x_vals.shape[0],)),
#             hist,
#             alpha=alpha,
#             color=color
#             )
#     # PLot expected value
#     ylims = ax.get_ylim()
#     rand_count_mean = np.mean(rand_frac)
#     ax.plot([rand_count_mean]*2, [0,0.75*ylims[1]], 'grey', lw=lw)
#     # plot location of actual assignment number
#     true_count = true_assoc_dict[bc_tax]
#     true_frac = true_count / n_pix
#     ax.plot([true_frac]*2, [0,0.75*ylims[1]], color=true_col, lw=true_lw)
#     ax.set_title(sci_name)

# %% codecell
# Plot as boxplot
dims=[5,3]
xlab_rotation=60
marker='*'
marker_size=10
text_dist=0.01
ft=8
ylimadj = 0.02
true_frac_llim = 0.02

# barcodes_int_order = [100,1,10000,1000,10]
barcodes_int_order = barcodes
dict_bc_sciname = dict(zip(barcodes, sci_names))

# create boxplot array
n_pix = np.sum([v for k, v in true_assoc_dict.items()])
sim_arr = [np.array(sim_dict[bc]) / n_pix for bc in barcodes_int_order]
# General plot
fig, ax = ip.general_plot(dims=dims, ft=ft)
# Plot simulation
boxplot = ax.boxplot(sim_arr, patch_artist=True, showfliers=False)
for m in boxplot['medians']:
    m.set_color('black')
for b in boxplot['boxes']:
    b.set_edgecolor('black')
    b.set_facecolor('white')
col_dict
# Plot measured value
ys = []
xlab = []
for i, bc_tax in enumerate(barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    color = col_dict[sci_name]
    true_count = true_assoc_dict[bc_tax]
    true_frac = true_count / n_pix
    _ = ax.plot(i+1, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals = np.array(sim_dict[bc_tax])
    sim_mean = np.mean(sim_vals)
    if true_count > sim_mean:
        r_ = sum(sim_vals > true_count)
    else:
        r_ = sum(sim_vals < true_count)
    p_ = r_ / n
    y_m = np.max(sim_vals/n_pix)
    y = y_m + text_dist
    ys.append(y)
    if true_frac < true_frac_llim:
        t = ''
    elif p_ > 0.001:
        t = str("p=" + str(p_))
    else:
        t = str("p<0.001")
    _ = ax.text(i+1, y, t, fontsize=ft, ha='center', rotation=xlab_rotation)
ax.set_xticklabels(xlab, rotation=xlab_rotation)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_spot_association'
# ip.save_png_pdf(out_bn)

# %% md

# ==============================================================================
# ## Work on classifier
# ==============================================================================

# Get zoom region

# %% codecell
# show rgb overlay on zoom
c = [2000, 0]  # corner
# c = [1400, 1400]  # corner
w = [1000,1000]  # height and width
# ulims = (1,1,0.45)

im_inches=10
stack_zoom = np.dstack(raws_max_norm_adj)[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
im_classif_rgb_zoom = im_classif_rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
im_classif_rgb_zoom.shape
ip.subplot_square_images([stack_zoom, im_classif_rgb_zoom], (1,2), clims=[[],[]], im_inches=im_inches)
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape//(im_inches*2))
# plt.show()
# plt.close()
# ip.plot_image(im_classif_rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:], im_inches=im_inches)

# %% codecell
# subset raw image
raws_sub = [im[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:] for im in raws_shift]
mask_sub = mask[c[0]:c[0]+w[0],c[1]:c[1]+w[1]]
raws_sub[0].shape


# %% md

# Cluster based on max value for each channel and see how that looks

# %% codecell
# smoothing
# sigma=3
sigma=10

raws_sub_smooth = []
for im in raws_sub:
    im_smooth = np.empty(im.shape)
    for i in range(im.shape[2]):
        im_smooth[:,:,i] = gaussian_filter(im[:,:,i], sigma=sigma)
    raws_sub_smooth.append(im_smooth)


# %% codecell
# Get max for each channel
las_max_sub = [np.max(im, axis=2) for im in raws_sub_smooth]
# Get arrays for pixels
las_max_sub_stack = np.dstack(las_max_sub)
las_max_sub_pix = las_max_sub_stack[mask_sub]
# Get full spectral pixels
stack_sub_smooth = np.dstack(raws_sub_smooth)
# Scaled pixels
spec_pix_sub = stack_sub_smooth[mask_sub,:]
pix_scaler_sub = StandardScaler().fit(spec_pix_sub)
# normalized scaled pixels
spec_pix_sub_scaled = pix_scaler_sub.transform(spec_pix_sub)
spec_pix_sub_norm = spec_pix_sub / np.max(spec_pix_sub, axis=1)[:,None]
pix_scaler_sub_norm = StandardScaler().fit(spec_pix_sub_norm)
spec_pix_sub_norm_scaled = pix_scaler_sub_norm.transform(spec_pix_sub_norm)
# norm scaled positive
spec_pix_sub_norm_scaled_pos = spec_pix_sub_norm_scaled - np.min(spec_pix_sub_norm_scaled, axis=1)[:,None]
# Get specific channels only
channels = [4,29,33,44,47]
# channels = [4,10,14,24,29,33,36,44,47]
stack_sub_smooth_chan = np.dstack([stack_sub_smooth[:,:,c] for c in channels])
pix_chan_sub = stack_sub_smooth_chan[mask_sub,:]
# remove gemella spectrum
ind_gem = np.where(np.array(sci_names) == 'Gemella')[0][0]
spec_gem = weights_max_norm[ind_gem]
scale_gem = 0.05
spec_pix_sub_gem = spec_pix_sub - spec_gem[None,:]*scale_gem
spec_pix_sub_gem[spec_pix_sub_gem < 0] = 0


# %% codecell
# Cluster
nclust=8

kmeans_lasmax = KMeans(n_clusters=nclust, random_state=42).fit_predict(las_max_sub_pix)
kmeans_spec = KMeans(n_clusters=nclust, random_state=42).fit_predict(spec_pix_sub)
kmeans_chan = KMeans(n_clusters=nclust, random_state=42).fit_predict(pix_chan_sub)

# %% codecell
# Project clusters back onto image
ul = 0.5
kmeans = kmeans_chan

plot_intensities = np.sum(np.dstack(raws_sub), axis=2)
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask_sub)
# for each index draw the color
tab20 = plt.get_cmap('Set1').colors
tab20_sort = tab20
# tab_sort = [tab20[i] for i in np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()]
# grey = tab_sort.pop(7)
# tab_sort = [grey] + tab_sort
labs = np.unique(kmeans, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_dict = {l:c for l, c in zip(labs_sort, tab20_sort)}
im_clust = np.zeros(mask_sub.shape + (len(tab20[0]),))
# im_clust = np.zeros(max_sub.shape + (len(eval(barcode_color.color.values[0])),))
for lab, i in zip(kmeans, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell
# Show the rgb vs the new projection
im_inches=20
ip.subplot_square_images([stack_zoom, im_clust], (1,2), im_inches=im_inches)


# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )

# %% codecell



n = 100
labs = np.unique(kmeans, return_counts=True)
len(kmeans)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (kmeans == l)
    # spec_ = spec_pix_sub_norm_scaled[bool,:]
    # spec_ = spec_pix_norm[bool,:]
    # spec_ = spec_pix_sub[bool,:]
    spec_ = spec_pix_sub_gem[bool,:]
    # spec_ = spec_pix_sub_norm_scaled_pos[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.5,'color':'r'})
    ax.set_title(l)
    # output_dir = config['output_dir'] + '/test_classif/mat_mul'
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # out_bn = output_dir + '/' + sn + '_' + l + '_spectra'
    # ip.save_png_pdf(out_bn)
    #

# %% codecell
# Look at lautropia mat mul classification
cluster = 6
spec_pix_sub_lau = spec_pix_sub_gem[kmeans == cluster, :]
# spec_pix_sub_lau = spec_pix_sub_norm_scaled[kmeans == cluster, :]
# spec_pix_sub_lau = spec_pix_sub_norm_scaled_pos[kmeans == cluster, :]

weights_pick = weights_sum_norm
input = spec_pix_sub_lau

weights_t = np.array(weights_pick).T
classif_mat = np.matmul(input, weights_t)
classifs_index = np.argmax(classif_mat, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell
# Remove possibilities if laser is present
las_max_sub_pix_lau = las_max_sub_pix[kmeans == cluster,:]
las_max_sub_pix_norm = las_max_sub_pix_lau / np.max(las_max_sub_pix_lau, axis=1)[:,None]
las_frac_thresh = [0.3,0.5,0.3]
las_max_present = las_max_sub_pix_norm > las_frac_thresh
present_filter = [
        [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,0]
        ]
(las_max_present == 0).any()
classif_adj_mat = np.matmul(las_max_present, np.array(present_filter))
classif_adj_mat_bool = classif_adj_mat == np.max(classif_adj_mat, axis=1)[:,None]
classif_mat_adj = classif_mat * classif_adj_mat_bool

# %% codecell


# %% codecell
# plot
cl_mat_mean = np.mean(classif_mat_adj, axis=0)
cl_mat_std = np.std(classif_mat_adj, axis=0)
fig, ax = ip.general_plot(dims=(10,5))
x = np.arange(len(sci_names))
ax.bar(x, cl_mat_mean, yerr=cl_mat_std)
ax.set_xticks(x)
_ = ax.set_xticklabels(sci_names, rotation=90)

# %% codecell
# show just one pixel classif
for i in range(100,110):
    cl_mat_ = classif_mat_adj[i,:]
    fig, ax = ip.general_plot(dims=(10,5))
    x = np.arange(len(sci_names))
    ax.bar(x, cl_mat_)
    ax.set_xticks(x)
    _ = ax.set_xticklabels(sci_names, rotation=90)
    plt.show()
    plt.close()


# %% codecell
# Measure distance to each spectrum
test_spec = input[0,:]
test_spec_norm = test_spec / np.max(test_spec)
dists = [fhc.euclid_dist_cumul_spec(test_spec_norm, s) for s in weights_max_norm]
fig, ax = ip.general_plot(dims=(10,5))
x = np.arange(len(sci_names))
ax.bar(x, dists)
ax.set_xticks(x)
_ = ax.set_xticklabels(sci_names, rotation=90)

# %% codecell
# Measure distance to each spectrum
test_spec = input[0,:]
test_spec_norm = test_spec / np.max(test_spec)
dists = [fhc.channel_cosine_intensity_5b_v2(test_spec_norm, s) for s in weights_max_norm]
fig, ax = ip.general_plot(dims=(10,5))
x = np.arange(len(sci_names))
ax.bar(x, dists)
ax.set_xticks(x)
_ = ax.set_xticklabels(sci_names, rotation=90)




# %% md

# ==============================================================================
# ## Redo classify with gemella subtraction
# ==============================================================================

# %% codecell
# remove gemella spectrum
ind_gem = np.where(np.array(sci_names) == 'Gemella')[0][0]
spec_gem = weights_max_norm[ind_gem]
scale_gem = 0.05
spec_pix_sub_gem = spec_pix_sub - spec_gem[None,:]*scale_gem
spec_pix_sub_gem[spec_pix_sub_gem < 0] = 0

# %% codecell
# Classify each pixel by picking the maximum weight
weights_pick = weights_sum_norm
input = spec_pix_sub_gem

weights_t = np.array(weights_pick).T
classif_mat = np.matmul(input, weights_t)
# classifs_index = np.argmax(classif_mat, axis=1)
# classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell


# %% codecell
# Remove possibilities if laser is present
# las_max_sub_pix_lau = las_max_sub_pix[kmeans == cluster,:]
las_max_sub_pix_norm = las_max_sub_pix / np.max(las_max_sub_pix, axis=1)[:,None]
las_frac_thresh = [0.4,0.6,0.3]
las_max_present = las_max_sub_pix_norm > las_frac_thresh
present_filter = np.array([
        [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,0]
        ])
classif_adj_mat = np.matmul(las_max_present, present_filter)
classif_adj_mat_bool = classif_adj_mat == np.max(classif_adj_mat, axis=1)[:,None]
classif_mat_adj = classif_mat * classif_adj_mat_bool

# %% codecell
# plot

cl_mat_mean = np.mean(classif_mat_adj[kmeans == cluster,:], axis=0)
cl_mat_std = np.std(classif_mat_adj[kmeans == cluster,:], axis=0)
fig, ax = ip.general_plot(dims=(10,5))
x = np.arange(len(sci_names))
ax.bar(x, cl_mat_mean, yerr=cl_mat_std)
ax.set_xticks(x)
_ = ax.set_xticklabels(sci_names, rotation=90)


# %% codecell
# Pick maximums
classifs_index = np.argmax(classif_mat_adj, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])


# %% codecell
# Project clusters back onto image
ul = 0.5

plot_intensities = np.sum(np.dstack(raws_sub), axis=2)
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask_sub)
# for each index draw the color
tab20 = plt.get_cmap('tab20').colors
# tab20_sort = tab20
tab_sort = [tab20[i] for i in np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()]
grey = tab_sort.pop(7)
# tab_sort = [grey] + tab_sort
labs = np.unique(classifs, return_counts=True)
labs
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_dict = {l:c for l, c in zip(labs_sort, tab_sort)}
im_clust = np.zeros(mask_sub.shape + (len(tab20[0]),))
# im_clust = np.zeros(max_sub.shape + (len(eval(barcode_color.color.values[0])),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell
# Show the rgb vs the new projection
im_inches=20
ip.subplot_square_images([stack_zoom, im_clust], (1,2), im_inches=im_inches)


# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )

# %% codecell



n = 100
labs = np.unique(classifs, return_counts=True)
len(classifs)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    # spec_ = spec_pix_sub_norm_scaled[bool,:]
    # spec_ = spec_pix_norm[bool,:]
    # spec_ = spec_pix_sub[bool,:]
    spec_ = spec_pix_sub_gem[bool,:]
    # spec_ = spec_pix_sub_norm_scaled_pos[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.5,'color':'r'})
    ax.set_title(l)
    # output_dir = config['output_dir'] + '/test_classif/mat_mul'
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # out_bn = output_dir + '/' + sn + '_' + l + '_spectra'
    # ip.save_png_pdf(out_bn)
    #


# %% md

# ==============================================================================
# ## Classify on clusters
# ==============================================================================

# %% codecell
# Get mat mul classifier for each cluster
kmeans = kmeans_chan
weights_pick = weights_sum_norm
in_spec = spec_pix_sub_gem

weights_t = np.array(weights_pick).T
clusters = np.unique(kmeans)
classifs = np.empty((kmeans.shape[0],), dtype='S20')
# classif_mat = np.zeros((kmeans.shape[0], weights_t.shape[1]))
for c in clusters:
    bool_cl = kmeans==c
    # Classify mean spectrum
    spec_cl = np.mean(in_spec[bool_cl,:], axis=0)
    classif_mat_cl = np.matmul(spec_cl, weights_t)
    # Presence filter
    las_max_cl = las_max_sub_pix[bool_cl]
    las_max_cl_norm = las_max_cl / np.max(las_max_cl, axis=1)[:,None]
    las_max_cl_present = las_max_cl_norm > las_frac_thresh
    cl_adj_mat = np.matmul(las_max_cl_present, present_filter)
    cl_adj_mat_bool = cl_adj_mat == np.max(cl_adj_mat, axis=1)[:,None]
    classif_mat_cl_adj = classif_mat_cl * cl_adj_mat_bool
    # Get classif
    classifs_index_cl = np.argmax(classif_mat_cl_adj, axis=1)
    classif = np.array([sci_names[i] for i in classifs_index_cl])
    classifs[bool_cl] = classif

# %% codecell
# Project clusters back onto image
ul = 0.5

plot_intensities = np.sum(np.dstack(raws_sub), axis=2)
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask_sub)
# for each index draw the color
tab20 = plt.get_cmap('tab20').colors
# tab20_sort = tab20
tab_sort = [tab20[i] for i in np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()]
grey = tab_sort.pop(7)
# tab_sort = [grey] + tab_sort
labs = np.unique(classifs, return_counts=True)
labs
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_dict = {l:c for l, c in zip(labs_sort, tab_sort)}
im_clust = np.zeros(mask_sub.shape + (len(tab20[0]),))
# im_clust = np.zeros(max_sub.shape + (len(eval(barcode_color.color.values[0])),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell
# Show the rgb vs the new projection
im_inches=20
ip.subplot_square_images([stack_zoom, im_clust], (1,2), im_inches=im_inches)


# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )

# %% codecell



n = 100
labs = np.unique(classifs, return_counts=True)
len(classifs)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    # spec_ = spec_pix_sub_norm_scaled[bool,:]
    # spec_ = spec_pix_norm[bool,:]
    # spec_ = spec_pix_sub[bool,:]
    spec_ = spec_pix_sub_gem[bool,:]
    # spec_ = spec_pix_sub_norm_scaled_pos[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.5,'color':'r'})
    ax.set_title(l)
    # output_dir = config['output_dir'] + '/test_classif/mat_mul'

# ==============================================================================
# ## Distance based classif
# ==============================================================================

# %% codecell
spec_used = spec_pix_sub_gem
weights_pick = weights_sum_norm

# Separate by laser
channel_indices = [0,23,43,57]
spec_las_norm = []
for i in range(len(channel_indices) - 1):
    ch0, ch1 = channel_indices[i:i+2]
    spec_las = spec_used[:,ch0:ch1]
    # NOrmalize lasers
    spec_las_norm.append(spec_las /(np.max(spec_las, axis=1)[:,None] + 1e-15))
# Generate bool present matrix and multiply
spec_las_norm_present = []
for i, sln in enumerate(spec_las_norm):
    bool_present_mat = np.ones(sln.shape)
    bool_present_mat *= las_max_present[:,i][:,None]
    spec_las_norm_present.append(sln * bool_present_mat)
# Separate ref by laser
weights_mat = np.array(weights_pick)
ref_las = []
ref_las_norm = []
for i in range(len(channel_indices) - 1):
    ch0, ch1 = channel_indices[i:i+2]
    ref_las_ = weights_mat[:,ch0:ch1]
    ref_las.append(ref_las_)
    # NOrmalize lasers
    ref_las_norm.append(ref_las_ / (np.max(ref_las_, axis=1)[:,None] + 1e-15))
# Generate bool present matrix and multiply
ref_max_present = present_filter.T
ref_las_norm_present = []
for i, sln in enumerate(ref_las_norm):
    bool_present_mat = np.ones(sln.shape)
    bool_present_mat *= ref_max_present[:,i][:,None]
    ref_las_norm_present.append(sln * bool_present_mat)

# %% codecell
# Get distance metrics for each ref for each pixel
rf = np.hstack(ref_las_norm_present)
sp = np.hstack(spec_las_norm_present)
dist_mat = np.zeros((sp.shape[0], rf.shape[0]))
for n, s in tqdm(enumerate(sp)):
    for m, r in enumerate(rf):
        # if n == 0 and m == 0: print(s.shape, r.)
        dist_mat[n,m] = fhc.euclid_dist_cumul_spec(r, s)

# %% codecell
# Multiply by present filter
adj_mat = classif_adj_mat_bool.copy()*1.
adj_mat[adj_mat == 0] = np.nan
dist_mat_adj = dist_mat * adj_mat
# dist_mat_adj = dist_mat * classif_adj_mat_bool

adj_mat[0,:]
dist_mat_adj[0,:]

# %% codecell
# Plot classifs for lautropia
cl_mat_mean = np.mean(dist_mat_adj[kmeans == cluster,:], axis=0)
cl_mat_std = np.std(dist_mat_adj[kmeans == cluster,:], axis=0)
fig, ax = ip.general_plot(dims=(10,5))
x = np.arange(len(sci_names))
ax.bar(x, cl_mat_mean, yerr=cl_mat_std)
ax.set_xticks(x)
_ = ax.set_xticklabels(sci_names, rotation=90)
ax.set_ylim(0,5)

# %% codecell
# Pick maximums
classifs_index = np.nanargmin(dist_mat_adj, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell
# Project clusters back onto image
ul = 0.5

plot_intensities = np.sum(np.dstack(raws_sub), axis=2)
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask_sub)
# for each index draw the color
tab20 = plt.get_cmap('tab20').colors
# tab20_sort = tab20
tab_sort = [tab20[i] for i in np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()]
grey = tab_sort.pop(7)
# tab_sort = [grey] + tab_sort
labs = np.unique(classifs, return_counts=True)
labs
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_dict = {l:c for l, c in zip(labs_sort, tab_sort)}
im_clust = np.zeros(mask_sub.shape + (len(tab20[0]),))
# im_clust = np.zeros(max_sub.shape + (len(eval(barcode_color.color.values[0])),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell
# Show the rgb vs the new projection
im_inches=20
ip.subplot_square_images([stack_zoom, im_clust], (1,2), im_inches=im_inches)


# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )

# %% codecell



n = 100
labs = np.unique(classifs, return_counts=True)
len(classifs)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    # spec_ = spec_pix_sub_norm_scaled[bool,:]
    # spec_ = spec_pix_norm[bool,:]
    # spec_ = spec_pix_sub[bool,:]
    spec_ = sp[bool,:]
    # spec_ = spec_pix_sub_norm_scaled_pos[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.5,'color':'r'})
    ax.set_title(l)


# %% md

# ==============================================================================
# ## Redo classify on whole image with gemella subtraction
# ==============================================================================

# %% codecell
# remove gemella spectrum
ind_gem = np.where(np.array(sci_names) == 'Gemella')[0][0]
spec_gem = weights_max_norm[ind_gem]
scale_gem = 0.05
spec_pix_gem = spec_pix - spec_gem[None,:]*scale_gem
spec_pix_gem[spec_pix_gem < 0] = 0

# %% codecell
# smoothing
sigma=3
# sigma=10

raws_smooth = []
for im in raws:
    im_smooth = np.empty(im.shape)
    for i in range(im.shape[2]):
        im_smooth[:,:,i] = gaussian_filter(im[:,:,i], sigma=sigma)
    raws_smooth.append(im_smooth)



# %% codecell
# Get max for each channel
las_max = [np.max(im, axis=2) for im in raws_smooth]
# Get arrays for pixels
las_max_stack = np.dstack(las_max)
las_max_pix = las_max_stack[mask]

# %% codecell
# Classify each pixel by picking the maximum weight
weights_pick = weights_sum_norm
input = spec_pix_gem

weights_t = np.array(weights_pick).T
classif_mat = np.matmul(input, weights_t)
# classifs_index = np.argmax(classif_mat, axis=1)
# classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell


# %% codecell
# Remove possibilities if laser is present
# las_max_pix_lau = las_max_pix[kmeans == cluster,:]
las_max_pix_norm = las_max_pix / np.max(las_max_pix, axis=1)[:,None]
las_frac_thresh = [0.3,0.4,0.3]  # Max value relative to normalized max for declaring a laser absent
las_max_present = las_max_pix_norm > las_frac_thresh
present_filter = np.array([
        [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,0]
        ])
classif_adj_mat = np.matmul(las_max_present, present_filter)
classif_adj_mat_bool = classif_adj_mat == np.max(classif_adj_mat, axis=1)[:,None]
classif_mat_adj = classif_mat * classif_adj_mat_bool


# %% codecell
# Pick maximums
classifs_index = np.argmax(classif_mat_adj, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell
# adjust intensities
plot_intensities = np.sum(np.dstack(raws), axis=2)
plot_intensities /= np.max(plot_intensities)

# gamma = 2  # Gamma correction
# plot_intensities = plot_intensities ** (1/gamma)
# fig, ax = ip.general_plot()
# ax.plot([0,1],[0,1])
# x = np.linspace(0,1,100)
# ax.plot(x, x**(1/gamma))
# ax.set_aspect('equal')

ul = 0.25  # white limit
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul




# %% codecell
# Project clusters back onto image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
tab20 = plt.get_cmap('tab20').colors
# tab20_sort = tab20
# tab_sort = [tab20[i] for i in np.arange(1,20,2).tolist() + np.arange(0,20,2).tolist()]
tab_sort = [tab20[i] for i in np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()]
grey = tab_sort.pop(7)
# tab_sort = [grey] + tab_sort
labs = np.unique(classifs, return_counts=True)
labs
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_dict = {l:c for l, c in zip(labs_sort, tab_sort)}
im_clust = np.zeros(mask.shape + (len(tab20[0]),))
# im_clust = np.zeros(max.shape + (len(eval(barcode_color.color.values[0])),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]


# %% codecell
# Show the a zoom of the rgb vs the new projection
c=[2000,2000]
d=[2000,2000]
im_inches=20
# rgb = np.dstack(raws_max_norm_adj)
ip.subplot_square_images([rgb[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], im_clust[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]], (1,2), im_inches=im_inches)


# %% codecell
# Show the rgb vs the new projection
im_inches=20
rgb = np.dstack(raws_max_norm_adj)
ip.subplot_square_images([rgb, im_clust], (1,2), im_inches=im_inches)
# output_dir = config['output_dir'] + '/test_classif/mat_mul'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# out_bn = output_dir + '/' + sn + '_rgb_plus_pixelclassif'
# ip.save_png_pdf(out_bn)
#


# %% codecell
# save the new projection
im_inches = 5
ip.plot_image(im_clust, im_inches=im_inches)
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_pixelclassif'
# ip.save_png_pdf(out_bn)

# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_pixelclassif_full_legend'
# ip.save_png_pdf(out_bn)


# %% codecell



n = 1000
labs = np.unique(classifs, return_counts=True)
len(classifs)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    # spec_ = spec_pix_norm_scaled[bool,:]
    # spec_ = spec_pix_norm[bool,:]
    # spec_ = spec_pix[bool,:]
    spec_ = spec_pix_gem[bool,:]
    # spec_ = spec_pix_norm_scaled_pos[bool,:]
    n_ = spec_.shape[0] if spec_.shape[0] < n else n
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.5,'color':'r'})
    ax.set_title(l)
    # output_dir = config['output_dir'] + '/test_classif/mat_mul'
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # out_bn = output_dir + '/' + sn + '_' + l + '_pixelclassif_full_spectra'
    # ip.save_png_pdf(out_bn)
    #

# %% md

# ==============================================================================
# ## Overlay new classif with MeGAFISH
# ==============================================================================

# %% codecell
# resize hiprfish images
dims = lrg.shape
ul_corner = shp_dff
hipr_classif_resize = resize_hipr(
        im_clust, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )


# %% codecell
# Plot spot overlay on classif
im_inches=10
ceil=1
size_min=0.1
marker_size=100
marker='.'
spot_col=(1,0,1)
edge_col='none'
linewidths=1.5

spot_props_shift = sss_props
fig, ax, cbar = ip.plot_image(
        hipr_classif_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
ref_pts = [c for c in spot_props_shift['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
ref_pts_arr = np.array(ref_pts)
# # Plot size as spot intensity
# spot_int = spot_props_shift.max_intensity.values.copy()
# spot_int -= np.min(spot_int)
# spot_int /= np.max(spot_int)
# spot_int[spot_int > ceil] = ceil
# spot_int /= ceil
# spot_int[spot_int < size_min] = size_min
# marker_size_arr = marker_size * spot_int
marker_size_arr=marker_size
ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
            marker=marker, s=marker_size_arr, color=spot_col,
            linewidths=linewidths, edgecolors=edge_col
            )
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spot_overlay_dot'
# ip.save_png_pdf(out_bn)


# %% codecell
# show raw spot on top of classif zoom
ul = 0.25
ll = 0.125
c=[1500,2500]
d=[750,750]
im_inches = 20
# zoom_coords=[c[0], c[0]+d[0], c[1], c[1]+d[1]]

hcr_zoom = hipr_classif_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
fig, ax, cbar = ip.plot_image(
        hcr_zoom, scalebar_resolution=mega_res, im_inches=im_inches
        )
spot_raw = raw_shift[0][:,:,1].copy()
spot_raw -= np.min(spot_raw)
spot_raw /= np.max(spot_raw)
spot_raw[spot_raw > ul] = ul
spot_raw[spot_raw < ll] = 0
spot_raw /= ul
spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
spot_raw_overlay[:,:,0] = spot_raw
spot_raw_overlay[:,:,2] = spot_raw
spot_raw_overlay[:,:,3] = spot_raw
sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(sro_zoom)
out_dir = config['output_dir'] + '/pixel_classif_test'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay_zoom'
# ip.save_png_pdf(out_bn, bbox_inches=False)

# %% codecell
# save raw spot on top of classif
im_inches = 20

fig, ax, cbar = ip.plot_image(
        hipr_classif_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
ax.imshow(spot_raw_overlay)
out_dir = config['output_dir'] + '/pixel_classif_test'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay'
# ip.save_png_pdf(out_bn)

# %% codecell
# show raw spot on top of raw cell zoom
hipr_sum_resize = resize_hipr(
        np.sum(stack, axis=2), hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )
# %% codecell
ul = 0.15
ll = 0.075
c=[1000,1000]
d=[1000,1000]
im_inches = 20
# zoom_coords=[c[0], c[0]+d[0], c[1], c[1]+d[1]]
hsr_zoom = hipr_sum_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
fig, ax, cbar = ip.plot_image(
        hsr_zoom, scalebar_resolution=mega_res, im_inches=im_inches,
        cmap='gray'
        )
spot_raw = raw_shift[0][:,:,1].copy()
spot_raw -= np.min(spot_raw)
spot_raw /= np.max(spot_raw)
spot_raw[spot_raw > ul] = ul
spot_raw[spot_raw < ll] = 0
spot_raw /= ul
spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
spot_raw_overlay[:,:,0] = spot_raw
spot_raw_overlay[:,:,2] = spot_raw
spot_raw_overlay[:,:,3] = spot_raw
sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(sro_zoom)
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_rgb_spotraw_overlay_zoom'
# ip.save_png_pdf(out_bn)



# %% codecell
# save raw spot on top of raw cell zoom
# c=[2000,5000]
# d=[2000,2000]
im_inches = 20
# zoom_coords=[c[0], c[0]+d[0], c[1], c[1]+d[1]]
# hipr_rgb_resize_zoom = hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
spot_raw = raw_shift[0][:,:,1].copy()
spot_raw -= np.min(spot_raw)
spot_raw /= np.max(spot_raw)
spot_raw[spot_raw > ul] = ul
spot_raw[spot_raw < ll] = 0
spot_raw /= ul
spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
spot_raw_overlay[:,:,0] = spot_raw
spot_raw_overlay[:,:,2] = spot_raw
spot_raw_overlay[:,:,3] = spot_raw
# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay)
out_dir = config['output_dir'] + '/pixel_classif_test'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_rgb_spotraw_overlay'
# ip.save_png_pdf(out_bn)

# %% md

# ==============================================================================
# ## Spatial association between spots and pixels
# ==============================================================================

# Calculate actual spatial assoc

# %% codecell
# Get nearest neighbor classifier for pixels
r_um = 2

dims = lrg.shape
ul_corner = shp_dff
spot_coords = ref_pts_arr
hipr_mask_resize = resize_hipr(
        mask*1, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )
pix_coords = np.argwhere(hipr_mask_resize)

r_pix = int(r_um / mega_res)

# nbrs = NearestNeighbors(radius=r_pix).fit(pix_coords)
# # Search in radius around each spot
# inds = nbrs.radius_neighbors(spot_coords, return_distance=False)
# inds



# %% codecell
# Get resized classifs list for pixel inds
barcodes_int = [int(bc) for bc in barcodes]
dict_sn_bc = dict(zip(sci_names, barcodes_int))
im_bc = np.zeros(max_sub.shape)
classifs_bc = [dict_sn_bc[sn] for sn in classifs]
for bc, i in zip(classifs_bc, pix_ind):
    x, y = i
    # col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_bc[x,y] = bc
hipr_bc_resize = resize_hipr(
        im_bc, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )

# %% codecell
# Get circle of pixels around spots
int_array = nb.types.int64[:]
@nb.njit()
def get_assocs(spot_coords, hipr_bc_resize, r_pix=50):
    # assocs_dict = defaultdict(lambda:0)
    assocs_dict = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
    for x, y in spot_coords:
        for i in range(2*r_pix):
            for j in range(2*r_pix):
                x_i = int(x - r_pix + i)
                y_j = int(y - r_pix + j)
                if ((x-x_i)**2 + (y-y_j)**2)**(1/2) <= r_pix:
                    # bc = hipr_bc_resize[y_j, x_i]
                    bc = np.int64(hipr_bc_resize[y_j, x_i])
                    # print(bc)
                    try:
                        assocs_dict[bc] += 1
                    except:
                        assocs_dict[bc] = 1
    return assocs_dict

# %% codecell
true_assoc_dict = get_assocs(spot_coords, hipr_bc_resize, r_pix=r_pix)
for bc in barcodes:
    try:
        _ = true_assoc_dict[bc]
    except:
        true_assoc_dict[bc] = 0
# values, counts = np.unique(assocs, return_counts=True)
# true_dict = dict(zip(values,counts))
true_assoc_dict

# %% codecell
# # Sum spatial assoc counts for each taxon
# values, counts = np.unique(classifs[inds], return_counts=True)
# true_dict = dict(zip(values,counts))

# %% md

# Simulate random distribution of spots

# %% codecell
# run knn on pixels
# inds_pix = nbrs.radius_neighbors(pix_coords, return_distance=False)
# iterate n times
n=1000
sim_dict = defaultdict(list)
for i in tqdm(range(n)):
    # Randomly select pixels
    i_sim = np.random.randint(0,pix_coords.shape[0], size=spot_coords.shape[0])
    # Generate counts for random selection
    sim_spot_coords = pix_coords[i_sim]
    sim_assoc_dict = get_assocs(sim_spot_coords, hipr_bc_resize, r_pix=r_pix)
    for bc in barcodes:
        try:
            _ = sim_assoc_dict[bc]
        except:
            sim_assoc_dict[bc] = 0
    for k, v in sim_assoc_dict.items():
        sim_dict[k].append(v)

# %% md

# Plot

# %% codecell
# Plot simulated values for random
# alpha=1
# true_col='k'
# true_lw=2
# lw=1
# dims=(3,2)
# ft=7
# nbins=100
# n_pix = np.sum([v for k, v in true_assoc_dict.items()])
# dict_bc_sciname = dict(zip(barcodes_int,sci_names))
# for bc_tax in barcodes_int:
#     sci_name = dict_bc_sciname[bc_tax]
#     color = col_dict[sci_name]
#     rand_counts = sim_dict[bc_tax]
#     # Get fraction of total spots
#     rand_frac = np.array(rand_counts) / n_pix
#     # plot distribution of spot assignment
#     nbins = np.unique(rand_frac).shape[0] // 4
#     bins = np.linspace(np.min(rand_frac), np.max(rand_frac), nbins)
#     # bins = np.arange(np.min(rand_frac), np.max(rand_frac))
#     # hist, bin_edges = np.histogram(rand_frac)
#     hist, bin_edges = np.histogram(rand_frac, bins=bins)
#     x_vals = ip.get_line_histogram_x_vals(bin_edges)
#     fig, ax = ip.general_plot(dims=dims, ft=ft, lw=lw)
#     ax.plot(x_vals, hist, color=color)
#     ax.fill_between(
#             x_vals,
#             np.zeros((x_vals.shape[0],)),
#             hist,
#             alpha=alpha,
#             color=color
#             )
#     # PLot expected value
#     ylims = ax.get_ylim()
#     rand_count_mean = np.mean(rand_frac)
#     ax.plot([rand_count_mean]*2, [0,0.75*ylims[1]], 'grey', lw=lw)
#     # plot location of actual assignment number
#     true_count = true_assoc_dict[bc_tax]
#     true_frac = true_count / n_pix
#     ax.plot([true_frac]*2, [0,0.75*ylims[1]], color=true_col, lw=true_lw)
#     ax.set_title(sci_name)

# %% codecell
# Plot as boxplot
dims=[5,3]
xlab_rotation=60
marker='*'
marker_size=10
text_dist=0.01
ft=8
ylimadj = 0.02
true_frac_llim = 0

# barcodes_int_order = [100,1,10000,1000,10]
barcodes_int_order = barcodes
dict_bc_sciname = dict(zip(barcodes, sci_names))

# create boxplot array
n_pix = np.sum([v for k, v in true_assoc_dict.items()])
sim_arr = [np.array(sim_dict[bc]) / n_pix for bc in barcodes_int_order]
# General plot
fig, ax = ip.general_plot(dims=dims, ft=ft)
# Plot simulation
boxplot = ax.boxplot(sim_arr, patch_artist=True, showfliers=False)
for m in boxplot['medians']:
    m.set_color('black')
for b in boxplot['boxes']:
    b.set_edgecolor('black')
    b.set_facecolor('white')
col_dict
# Plot measured value
ys = []
xlab = []
for i, bc_tax in enumerate(barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    try:
        color = col_dict[sci_name]
    except:
        continue
    true_count = true_assoc_dict[bc_tax]
    true_frac = true_count / n_pix
    _ = ax.plot(i+1, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals = np.array(sim_dict[bc_tax])
    sim_mean = np.mean(sim_vals)
    if true_count > sim_mean:
        r_ = sum(sim_vals > true_count)
    else:
        r_ = sum(sim_vals < true_count)
    p_ = r_ / n
    y_m = np.max(sim_vals/n_pix)
    y = y_m + text_dist
    ys.append(y)
    if true_frac < true_frac_llim:
        t = ''
    elif p_ > 0.001:
        t = str("p=" + str(p_))
    else:
        t = str("p<0.001")
    _ = ax.text(i+1, y, t, fontsize=ft, ha='center', rotation=xlab_rotation)
ax.set_xticklabels(xlab, rotation=xlab_rotation)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_spot_association'
# ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## assign identities to segmentation
# ==============================================================================

# %% codecell
# Show zoom of classif overlayed with seg
# Load seg
hipr_seg_fmt = config['hipr_dir'] + '/' + config_hipr['output_dir'] + '/' + config_hipr['seg_fmt']
hipr_seg = np.load(hipr_seg_fmt.format(sample_name=sn))
# %% codecell
# resize
dims = lrg.shape
ul_corner = shp_dff
spot_coords = ref_pts_arr
hipr_seg_resize = resize_hipr(
        hipr_seg, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )

# %% codecell
# overlay
c=[2000,2500]
d=[750,750]
im_inches = 20
# zoom_coords=[c[0], c[0]+d[0], c[1], c[1]+d[1]]

hcr_zoom = hipr_classif_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
fig, ax, cbar = ip.plot_image(
        hcr_zoom, scalebar_resolution=mega_res, im_inches=im_inches
        )
hsr_zoom = hipr_seg_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
ip.plot_seg_outline(ax, hsr_zoom, col=(1,1,1))
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay_zoom'
# ip.save_png_pdf(out_bn)

# %% codecell
# Sub segment the image based on classifier'
hipr_seg_resize_props = sf.measure_regionprops(hipr_seg_resize)
bboxes = hipr_seg_resize_props.bbox
labels = hipr_seg_resize_props.label

hipr_reseg = sf.re_segment_with_classif(
        hipr_seg_resize, hipr_bc_resize, bboxes, labels
        )

# %% codecell
c=[2000,2500]
d=[750,750]
im_inches = 20
# zoom_coords=[c[0], c[0]+d[0], c[1], c[1]+d[1]]

hcr_zoom = hipr_classif_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
fig, ax, cbar = ip.plot_image(
        hcr_zoom, scalebar_resolution=mega_res, im_inches=im_inches
        )
hsr_zoom = hipr_reseg[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
ip.plot_seg_outline(ax, hsr_zoom, col=(1,1,1))


# %% codecell
# Get regionprops for new seg
hipr_reseg_props = sf.measure_regionprops(hipr_reseg, raw=hipr_bc_resize)

# %% md

# ==============================================================================
# ## Nearest neighbor cells to spots spatial assoc
# ==============================================================================


# %% codecell
# Get nearest neighbors
h_max_um = 10
h_step_um = 0.5
n_neighbors=1

# Convert um measure to pixels
h_max_pix = h_max_um / config_mega['resolution']
h_step_pix = h_step_um / config_mega['resolution']
# Get a range of distances to measure how many associations
hs = np.arange(0, h_max_pix, h_step_pix)
# Save the locations of each set of species
barcodes_unq = hipr_reseg_props.max_intensity.unique().tolist()
barcodes_unq.remove(0)
dict_cell_coords = {}
for bc in barcodes_unq:
    bc_bool = hipr_reseg_props.max_intensity == bc
    reseg_centr = hipr_reseg_props.loc[bc_bool, 'centroid'].values
    reseg_coords = np.array([list(c) for c in reseg_centr])
    dict_cell_coords[bc] = reseg_coords


# %% codecell
# Measure the sample
# get the spot coordinates and create a nneighbors object from them
spot_coords_float = np.array([list(c) for c in spot_props_shift.centroid.values])
nbrs = NearestNeighbors(n_neighbors=n_neighbors)
nbrs.fit(spot_coords_float)
# initialize a dict to save the count values for each distance
dict_h = {}
for bc in barcodes_unq:
    reseg_coords = dict_cell_coords[bc]
    dists, inds = nbrs.kneighbors(reseg_coords)
    dict_h[bc] = []
    for h in hs:
        count = np.sum(dists < h)
        dict_h[bc].append(count)


# %% codecell
# simulate random distribution of spots
n = 1000

sim_list = []
for i in tqdm(range(n)):
    # Get random pixels
    i_sim = np.random.randint(
            0, pix_coords.shape[0], size=spot_coords_float.shape[0]
            )
    sim_spot_coords = pix_coords[i_sim]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(sim_spot_coords)
    list_bc = []
    for bc in barcodes_unq:
        reseg_coords = dict_cell_coords[bc]
        dists, inds = nbrs.kneighbors(reseg_coords)
        list_h = []
        for h in hs:
            count = np.sum(dists < h)
            list_h.append(count)
        list_bc.append(list_h)
    sim_list.append(list_bc)

sim_arr = np.array(sim_list)
sim_arr.shape

# %% codecell
# Plot simulation min max vs sample
dims=(5,3)
sim_lw=1
sim_col = 'k'
sim_alpha=0.5
sample_color='r'
sample_lw=2

for i, bc in enumerate(barcodes_unq[1:]):
    fig, ax = ip.general_plot(dims=dims)
    sim_bc = sim_arr[:,i,:]
    sim_bc_ll = np.min(sim_bc, axis=0)
    sim_bc_ul = np.max(sim_bc, axis=0)
    ax.plot(hs, sim_bc_ll, color=sim_col, lw=sim_lw)
    ax.plot(hs, sim_bc_ul, color=sim_col, lw=sim_lw)
    ax.plot(hs, dict_h[bc], lw=sample_lw, color=sample_color)
    print(dict_bc_sciname[bc])
    plt.show()
    plt.close()


# %% codecell
# Plot simulation array vs sample
dims=(5,3)
sim_lw=0.5
sim_col = 'k'
sim_alpha=0.1
sample_color='r'
sample_lw=2

len(spot_coords)
for i, bc in enumerate(barcodes_unq[1:]):
    fig, ax = ip.general_plot(dims=dims)
    sim_bc = sim_arr[:,i,:]
    for h in sim_bc:
        ax.plot(hs, h, lw=sim_lw, alpha=sim_alpha, color=sim_col)
    ax.plot(hs, dict_h[bc], lw=sample_lw, color=sample_color)
    print(dict_bc_sciname[bc])
    plt.show()
    plt.close()


# %% codecell
# Plot as boxplot for a given h
h = 2

h_um = hs[h]*config_mega['resolution']
dims=[2.75,1.5]
xlab_rotation=45
pval_rotation=60
marker='*'
marker_size=10
text_dist=0.1
ft=7
ylimadj = 0.1
true_frac_llim = 0
line_col = 'k'
box_col = 'w'

# barcodes_int_order = [100,1,10000,1000,10]
#

# Order barcodes by p value
pvals_pos, pvals_neg = [], []
bc_pos, bc_neg = [], []
for i, bc_tax in enumerate(barcodes_unq):
    true_count = dict_h[bc_tax][h]
    n_cells = dict_cell_coords[bc_tax].shape[0]
    true_frac = true_count / n_cells
    sim_vals = sim_arr[:,i,h] / n_cells
    sim_mean = np.mean(sim_vals)
    if true_frac > sim_mean:
        r_ = sum(sim_vals > true_frac)
        pvals_pos.append(r_ / n)
        bc_pos.append(bc_tax)
    else:
        r_ = sum(sim_vals < true_frac)
        pvals_neg.append(r_ / n)
        bc_neg.append(bc_tax)

barcodes_pos_order = [x for _, x in sorted(zip(pvals_pos, bc_pos), reverse=False)]
barcodes_neg_order = [x for _, x in sorted(zip(pvals_neg, bc_neg), reverse=True)]
barcodes_int_order = barcodes_pos_order + barcodes_neg_order

# Order the indices of the barcodes
ind_order = []
for bc in barcodes_int_order:
    ind = np.argwhere(np.array(barcodes_unq) == bc)[0][0]
    ind_order.append(ind)

barcodes_unq[12]
dict_bc_sciname = dict(zip(barcodes, sci_names))

# create boxplot array
# n_pix = np.sum([v for k, v in true_assoc_dict.items()])
# sim_arr = [np.array(sim_dict[bc]) / n_pix for bc in barcodes_int_order]
sim_arr_h = []
for i, bc in zip(ind_order, barcodes_int_order):
    n_cells = dict_cell_coords[bc].shape[0]
    sim_arr_h.append(sim_arr[:,i,h]/n_cells)
# General plot
fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
# Plot simulation
boxplot = ax.boxplot(
        sim_arr_h, patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=box_col, color=line_col),
        capprops=dict(color=line_col),
        whiskerprops=dict(color=line_col),
        medianprops=dict(color=line_col),
      )
# for m in boxplot['medians']:
#     m.set_color(line_col)
# for b in boxplot['boxes']:
#     b.set_edgecolor(line_col)
#     b.set_facecolor(box_col)
col_dict
# Plot measured value
ys = []
xlab = []
x = 1
for i, bc_tax in zip(ind_order, barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    try:
        color = col_dict[sci_name]
    except:
        continue
    true_count = dict_h[bc_tax][h]
    n_cells = dict_cell_coords[bc_tax].shape[0]
    true_frac = true_count / n_cells
    _ = ax.plot(x, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals = sim_arr[:,i,h] / n_cells
    sim_mean = np.mean(sim_vals)
    if true_frac > sim_mean:
        # number of simulations with value greater than observed
        r_ = sum(sim_vals > true_frac)
    else:
        # number of simulations with value less than observed
        r_ = sum(sim_vals < true_frac)
    # P value
    p_ = r_ / n
    # Get text location
    q1,q3 = np.quantile(sim_vals, [0.25,0.75])
    q4 = q3 + 1.5 * (q3 - q1)
    # y_m = np.max(sim_vals)
    # y = y_m if y_m > true_frac else true_frac
    y = q4 if q4 > true_frac else true_frac
    y += text_dist
    ys.append(y)
    if true_frac < true_frac_llim:
        t = ''
    elif (p_ > 0.05):
        t = ''
    elif (p_ > 0.001) and (p_ <= 0.05):
        t = str("p=" + str(p_))
    else:
        t = str("p<0.001")
    _ = ax.text(x, y, t, fontsize=ft, ha='left',va='bottom', rotation=pval_rotation, rotation_mode='anchor',
            color=line_col)
    x+=1
ax.set_xticklabels([], rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
# ax.set_xticklabels(xlab, rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
ax.tick_params(axis='x',direction='out')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ylims = ax.get_ylim()
ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
print(str(h_um) + ' micrometers')
out_dir = config['output_dir'] + '/spatial_assoc'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_spot_association_frac_cells_dist_1um'
# ip.save_png_pdf(out_bn)



# %% md

# ==============================================================================
# ## Nearest neighbor spots to cells spatial assoc
# ==============================================================================

# %% codecell
# Do nearest neighbor spots to cells
n_neighbors=2

dict_cell_nbrs = {}
for bc in barcodes_int_order:
    reseg_coords = dict_cell_coords[bc]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    dict_cell_nbrs[bc] = nbrs.fit(reseg_coords)

# %% codecell
# Measure the sample
# get the spot coordinates and create a nneighbors object from them
# initialize a dict to save the count values for each distance
dict_h = {}
for bc in barcodes_int_order:
    nbrs = dict_cell_nbrs[bc]
    dists, inds = nbrs.kneighbors(spot_coords_float)
    dict_h[bc] = []
    for h in hs:
        count = np.sum(dists < h)
        dict_h[bc].append(count)

# %% codecell
# simulate random distribution of spots
n = 1000

sim_list = []
for i in tqdm(range(n)):
    # Get random pixels
    i_sim = np.random.randint(
            0, pix_coords.shape[0], size=spot_coords_float.shape[0]
            )
    sim_spot_coords = pix_coords[i_sim]
    list_bc = []
    for bc in barcodes_int_order:
        nbrs = dict_cell_nbrs[bc]
        dists, inds = nbrs.kneighbors(sim_spot_coords)
        list_h = []
        for h in hs:
            count = np.sum(dists < h)
            list_h.append(count)
        list_bc.append(list_h)
    sim_list.append(list_bc)

sim_arr = np.array(sim_list)
sim_arr.shape

# %% codecell
# Plot simulation array vs sample
# dims=(5,3)
dims=[1.,0.75]
sim_lw=0.5
sim_col = 'k'
sim_alpha=0.1
# sample_color='r'
sample_lw=2
save_list = ['Veillonella','Leptotrichia','Fusobacterium']
ft = 6

hs_um = hs*config_mega['resolution']
len(spot_coords)
for i, bc in enumerate(barcodes_int_order):
    sci_name = dict_bc_sciname[bc]
    if sci_name in save_list:
        color = col_dict[sci_name]
        fig, ax = ip.general_plot(dims=dims, ft=ft)
        sim_bc = sim_arr[:,i,:]
        for h in sim_bc:
            ax.plot(hs_um, h, lw=sim_lw, alpha=sim_alpha, color=sim_col)
        ax.plot(hs_um, dict_h[bc], lw=sample_lw, color=color)
        # ax.plot(hs_um, dict_h[bc], lw=sample_lw, color=sample_color)
        # if not i == 3:
        ax.set_xticks(ticks=[0,2,4,6,8], labels=[])
        ax.set_yticks(ticks=[0,50,100,150], labels=[])
        print(dict_bc_sciname[bc])
        out_dir = config['output_dir'] + '/spatial_assoc'
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        out_bn = out_dir + '/' + sn + '_cell_association_lag_plot_' + sci_name
        # ip.save_png_pdf(out_bn)
        print('saved')

    plt.show()
    plt.close()


# %% codecell
# Plot as boxplot for a given h
h = 5

hs_um[h]
dims=[2.75,1.5]
xlab_rotation=45
pval_rotation=60
marker='*'
marker_size=10
text_dist=0.1
ft=7
ylimadj = 0.1
true_frac_llim = 0
line_col = 'k'
box_col = 'w'

# barcodes_int_order = [100,1,10000,1000,10]
# barcodes_int_order = barcodes_unq[1:]
dict_bc_sciname = dict(zip(barcodes, sci_names))

# create boxplot array
n_pix = np.sum([v for k, v in true_assoc_dict.items()])
# sim_arr = [np.array(sim_dict[bc]) / n_pix for bc in barcodes_int_order]
sim_arr_h = []
n_spots = spot_coords_float.shape[0]
for i, bc in enumerate(barcodes_int_order):
    n_cells = dict_cell_coords[bc].shape[0]
    sim_arr_h.append(sim_arr[:,i,h]/n_spots)
# General plot
fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
# Plot simulation
boxplot = ax.boxplot(
        sim_arr_h, patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=box_col, color=line_col),
        capprops=dict(color=line_col),
        whiskerprops=dict(color=line_col),
        medianprops=dict(color=line_col),
      )
# for m in boxplot['medians']:
#     m.set_color(line_col)
# for b in boxplot['boxes']:
#     b.set_edgecolor(line_col)
#     b.set_facecolor(box_col)
col_dict
# Plot measured value

n_spots = spot_coords_float.shape[0]
ys = []
xlab = []
x = 1
for i, bc_tax in enumerate(barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    try:
        color = col_dict[sci_name]
    except:
        continue
    true_count = dict_h[bc_tax][h]
    # n_cells = dict_cell_coords[bc_tax].shape[0]
    true_frac = true_count / n_spots
    _ = ax.plot(i+1, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals = sim_arr_h[i]
    sim_mean = np.mean(sim_vals)
    if true_frac > sim_mean:
        r_ = sum(sim_vals > true_frac)
    else:
        r_ = sum(sim_vals < true_frac)
    p_ = r_ / n
    # Get text location
    q1,q3 = np.quantile(sim_vals, [0.25,0.75])
    q4 = q3 + 1.5 * (q3 - q1)
    # y_m = np.max(sim_vals)
    # y = y_m if y_m > true_frac else true_frac
    y = q4 if q4 > true_frac else true_frac
    y += text_dist
    ys.append(y)
    if true_frac < true_frac_llim:
        t = ''
    elif (p_ > 0.05):
        t = ''
    elif (p_ > 0.001) and (p_ <= 0.05):
        t = str("p=" + str(p_))
    else:
        t = str("p<0.001")
    _ = ax.text(x, y, t, fontsize=ft, ha='left',va='bottom',
            rotation=pval_rotation, rotation_mode='anchor', color=line_col)
    x+=1
# ax.set_xticks(labels=xlab, ticks=np.arange(len(barcodes_int_order))+1)
# # ax.tick_params(axis='x',direction='in')
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# # ax.set_xticklabels(xlab, rotation=xlab_rotation)
ax.set_xticklabels([], rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
# ax.set_xticklabels(xlab, rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
ax.tick_params(axis='x',direction='in')
# ax.tick_params(axis='x',direction='out')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ylims = ax.get_ylim()
ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
print(str(hs_um[h]) + ' micrometers')
out_dir = config['output_dir'] + '/spatial_assoc'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_cell_association_frac_spots_dist_2_5um'
# ip.save_png_pdf(out_bn)


# ==============================================================================
# ## nearest neighbor histograms
# ==============================================================================

# %% codecell
# For each taxon
n_neighbors=1

dict_cell_nbrs = {}
for bc in barcodes_int_order:
    reseg_coords = dict_cell_coords[bc]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(reseg_coords)
    dists, _ = nbrs.kneighbors(spot_coords)
    dict_bc_dists[bc] = dists

# %% codecell
dims=(3,3)
ft=12
line_col='k'
lw=3
alpha=0.75
bins=20
xlims=(0,5)

fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
for bc in barcodes_int_order:
    sci_name = dict_bc_sciname[bc]
    color = col_dict[sci_name]
    dists = dict_bc_dists[bc]
    # hist, bin_edges = np.histogram(dists, bins=bins)
    hist, bin_edges = np.histogram(dists, bins=bins, density=True)
    x = ip.get_line_histogram_x_vals(bin_edges) * config_mega['resolution']
    ax.plot(x, hist, color=color, lw=lw, alpha=alpha, label=sci_name)
ax.set_xlim(xlims[0],xlims[1])
# ax.legend()

output_dir = config['output_dir'] + '/nn_histogram'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_nn_histogram'
# ip.save_png_pdf(out_bn)

hist.sum()


# %% md

# Random simulation

# %% codecell
spot_coords = ref_pts_arr
spot_coords_float = np.array([list(c) for c in spot_props_shift.centroid.values])
hipr_mask_resize = resize_hipr(
        mask*1, hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
        )
pix_coords = np.argwhere(hipr_mask_resize)

r_pix = int(r_um / mega_res)

# %% codecell
# Simulate random cells and random spots
n=1000

cell_coords_tup = hipr_reseg_props.loc[:,'centroid'].values
cell_coords = np.array([list(c) for c in cell_coords_tup])
cell_bc = hipr_reseg_props.max_intensity.astype(int).tolist()

dict_bc_dists_sim = defaultdict(list)
for i in tqdm(range(n)):
    # Randomize spot locations
    i_sim = np.random.randint(
            0, pix_coords.shape[0], size=spot_coords_float.shape[0]
            )
    sim_spot_coords = pix_coords[i_sim]
    # # Randomize cell labels
    # c_sim = np.random.randint(
    #         0, hipr_reseg_props.shape[0], size=hipr_reseg_props.shape[0]
    #         )
    # bc_sim = np.array([x for _, x in sorted(zip(c_sim, cell_bc))])
    for bc in barcodes_int_order:
        # Get cell coords for taxon
        bool_bc = np.array(cell_bc) == bc
        # bool_bc = bc_sim == bc
        tax_centroid = cell_coords[bool_bc,:]
        # Get nearest neighbor cell distance for each spot
        nbrs = NearestNeighbors(n_neighbors=1).fit(tax_centroid)
        dists, _ = nbrs.kneighbors(sim_spot_coords)
        dict_bc_dists_sim[bc].append(dists)

# %% codecell
# plot simulation
dims=(1.5,0.75)
ft=6
line_col='k'
lw=3
alpha=0.75
bins=20
xlims=(0,5)
ylims=(0,1)
color_sim = 'k'
alpha_sim=0.15
lw_sim=0.5
bin_width_um=0.5

max_dist = np.max([d for k, d in dict_bc_dists_sim.items()])
max_dist_um = max_dist * config_mega['resolution']
bins = np.arange(0, max_dist_um,bin_width_um)

for bc in barcodes_int_order:
    fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
    # PLot simulation
    sim_dists = np.array(dict_bc_dists_sim[bc]) * config_mega['resolution']
    for i in tqdm(range(n)):
        hist, bin_edges = np.histogram(sim_dists[i,:], bins=bins, density=True)
        # hist /= sum(hist)
        x = ip.get_line_histogram_x_vals(bin_edges)
        ax.plot(x, hist, color=color_sim, lw=lw_sim, alpha=alpha_sim)
    # PLot observed
    sci_name = dict_bc_sciname[bc]
    color = col_dict[sci_name]
    dists = dict_bc_dists[bc] * config_mega['resolution']
    # hist, bin_edges = np.histogram(dists, bins=bins)
    hist, bin_edges = np.histogram(dists, bins=bins, density=True)
    # hist /= sum(hist)
    x = ip.get_line_histogram_x_vals(bin_edges)
    ax.plot(x, hist, color=color, lw=lw, alpha=alpha, label=sci_name)
    # adjust plot
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_xlim(xlims[0],xlims[1])
    print(sci_name)
    output_dir = config['output_dir'] + '/nn_histogram'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    out_bn = output_dir + '/' + sn + '_nn_histogram_' + sci_name
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()


# %% md

# ==============================================================================
# ## segmentatino Nearest neighbor spots to cells spatial assoc
# ==============================================================================

# %% codecell
# Fraction of spots associated
r_um = 0.5

meas_vals = []
sim_vals = []
for bc in barcodes_int_order:
    # Get simulated fraction within radius of cell
    sim_dists = dict_bc_dists_sim[bc]
    sim_dists_um = np.array(sim_dists) * config_mega['resolution']
    bool_sim_rad = sim_dists_um < r_um
    sim_rad_counts = np.sum(bool_sim_rad, axis=1)
    sim_rad_frac = sim_rad_counts
    sim_vals.append(sim_rad_frac)
    # Get measured fraction
    dists_um = dict_bc_dists[bc] * config_mega['resolution']
    dists_um.shape
    bool_rad = dists_um < r_um
    rad_counts = np.sum(bool_rad)
    rad_frac = rad_counts
    meas_vals.append(rad_frac)

sim_vals = np.array(sim_vals)[:,:,0]
sim_frac = sim_vals / spot_coords.shape[0]
meas_vals = np.array(meas_vals)
meas_frac = meas_vals / spot_coords.shape[0]


# %% codecell
# Plot fraction of spots
dims=[2.3,1.25]
xlab_rotation=45
pval_rotation=60
marker='.'
marker_size=10
text_dist=0.1
ft=7
ylimadj = 0.1
true_frac_llim = 0
line_col = 'k'
box_line_col = (0.5,0.5,0.5)
box_col = 'w'

fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
# Plot simulation
boxplot = ax.boxplot(
        sim_frac.T, patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=box_col, color=box_line_col),
        capprops=dict(color=box_line_col),
        whiskerprops=dict(color=box_line_col),
        medianprops=dict(color=box_line_col),
      )
# for m in boxplot['medians']:
#     m.set_color(line_col)
# for b in boxplot['boxes']:
#     b.set_edgecolor(line_col)
#     b.set_facecolor(box_col)
col_dict
# Plot measured value
ys = []
xlab = []
x = 1
for i, bc_tax in enumerate(barcodes_int_order):
# for i, bc_tax in zip(ind_order, barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    try:
        color = col_dict[sci_name]
    except:
        continue
    true_frac = meas_frac[i]
    # true_frac = true_count / n_cells
    _ = ax.plot(x, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals_i = sim_vals[i,:]
    # sim_vals = sim_arr[:,i,h] / n_cells
    sim_mean = np.mean(sim_vals)
    if true_frac > sim_mean:
        # number of simulations with value greater than observed
        r_ = sum(sim_vals_i > true_frac)
    else:
        # number of simulations with value less than observed
        r_ = sum(sim_vals_i < true_frac)
    # P value
    p_ = r_ / n
    # Get text location
    q1,q3 = np.quantile(sim_vals, [0.25,0.75])
    q4 = q3 + 1.5 * (q3 - q1)
    # y_m = np.max(sim_vals)
    # y = y_m if y_m > true_frac else true_frac
    y = q4 if q4 > true_frac else true_frac
    y += text_dist
    ys.append(y)
    # if true_frac < true_frac_llim:
    #     t = ''
    # elif (p_ > 0.05):
    #     t = ''
    # elif (p_ > 0.001) and (p_ <= 0.05):
    #     t = str("p=" + str(p_))
    # else:
    #     t = str("p<0.001")
    # _ = ax.text(x, y, t, fontsize=ft, ha='left',va='bottom', rotation=pval_rotation, rotation_mode='anchor',
    #         color=line_col)
    x+=1
ax.set_xticklabels([], rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
# ax.set_xticklabels(xlab, rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
ax.tick_params(axis='x',direction='out')
# ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

ylims = ax.get_ylim()
# ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
# out_dir = config['output_dir'] + '/spatial_assoc'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_seg_nn_frac_spot_association_0_5um'
# ip.save_png_pdf(out_bn)

# %% codecell
# Z-score number of spots associated with taxon
mu = np.mean(sim_vals, axis=1)
sig = np.std(sim_vals, axis=1)
sim_z = (sim_vals - mu[:,None]) / sig[:,None]
meas_z = (meas_vals - mu) / sig


# %% codecell
# Plot z score
dims=[2.5,1]
xlab_rotation=45
pval_rotation=60
marker='.'
marker_size=10
text_dist=0.1
ft=7
ylimadj = 0.1
true_frac_llim = 0
line_col = 'k'
box_line_col = (0.5,0.5,0.5)
box_col = 'w'
yticklength=2

fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
# Plot simulation
boxplot = ax.boxplot(
        sim_z.T, patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=box_col, color=box_line_col),
        capprops=dict(color=box_line_col),
        whiskerprops=dict(color=box_line_col),
        medianprops=dict(color=box_line_col),
      )
# for m in boxplot['medians']:
#     m.set_color(line_col)
# for b in boxplot['boxes']:
#     b.set_edgecolor(line_col)
#     b.set_facecolor(box_col)
col_dict
# Plot measured value
ys = []
xlab = []
x = 1
for i, bc_tax in enumerate(barcodes_int_order):
# for i, bc_tax in zip(ind_order, barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    try:
        color = col_dict[sci_name]
    except:
        continue
    true_frac = meas_z[i]
    # true_frac = true_count / n_cells
    _ = ax.plot(x, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals_i = sim_vals[i,:]
    # sim_vals = sim_arr[:,i,h] / n_cells
    sim_mean = np.mean(sim_vals)
    if true_frac > sim_mean:
        # number of simulations with value greater than observed
        r_ = sum(sim_vals_i > true_frac)
    else:
        # number of simulations with value less than observed
        r_ = sum(sim_vals_i < true_frac)
    # P value
    p_ = r_ / n
    # Get text location
    q1,q3 = np.quantile(sim_vals, [0.25,0.75])
    q4 = q3 + 1.5 * (q3 - q1)
    # y_m = np.max(sim_vals)
    # y = y_m if y_m > true_frac else true_frac
    y = q4 if q4 > true_frac else true_frac
    y += text_dist
    ys.append(y)
    # if true_frac < true_frac_llim:
    #     t = ''
    # elif (p_ > 0.05):
    #     t = ''
    # elif (p_ > 0.001) and (p_ <= 0.05):
    #     t = str("p=" + str(p_))
    # else:
    #     t = str("p<0.001")
    # _ = ax.text(x, y, t, fontsize=ft, ha='left',va='bottom', rotation=pval_rotation, rotation_mode='anchor',
    #         color=line_col)
    x+=1
# ax.set_xticklabels([], rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
# ax.set_xticklabels(xlab, rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
# ax.tick_params(axis='x',direction='out')
ax.set_xticks([])
ax.tick_params(axis='y', length=yticklength)
ax.set_yticks(ticks=[-10,0,10,20], labels=[])
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['right'].set_color('none')

# ylims = ax.get_ylim()
# ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
out_dir = config['output_dir'] + '/spatial_assoc'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_seg_nn_zscore_association_0_5um'
# ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Bar plot fraction of spots and cells assoc
# ==============================================================================

# %% codecell
# Frac spots assoc with taxon
# dims=[2.1,0.7]
ft=6
line_col = 'k'
width=0.4
dims=[2.5,0.6]
yticklength=2

sci_name_order = [dict_bc_sciname[bc] for bc in barcodes_int_order]
color_order = [col_dict[sc] for sc in sci_name_order]

fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
ax.bar(
        np.arange(meas_frac.shape[0]),
        meas_frac,
        width=width,
        color=color_order,
        edgecolor=line_col
        )

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xticks([])
ax.set_yticks(ticks=[0,0.2,0.4], labels=[])
ax.tick_params(axis='y', length=yticklength)

out_dir = config['output_dir'] + '/spatial_assoc'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_bar_seg_nn_frac_spot_association_0_5um'
# ip.save_png_pdf(out_bn)


# %% codecell
tax_counts = np.array([
        hipr_reseg_props[hipr_reseg_props.max_intensity == bc].shape[0]
        for bc in barcodes_int_order
        ])

meas_frac_cell = meas_vals / tax_counts

# %% codecell
# Frac spots assoc with taxon
dims=[2.5,0.6]
yticklength=2
ft=6
line_col = 'k'
width=0.4


fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
ax.bar(
        np.arange(meas_frac_cell.shape[0]),
        meas_frac_cell,
        width=width,
        color=color_order,
        edgecolor=line_col
        )
ax.set_xticks([])
ax.set_yticks(ticks=[0,0.3,0.6], labels=[])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(axis='y', length=yticklength)

out_dir = config['output_dir'] + '/spatial_assoc'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_bar_seg_nn_frac_cell_association_0_5um'
# ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Project only the veillonella
# ==============================================================================

# %% codecell
# adjust intensities
plot_intensities_veil = np.sum(np.dstack(raws), axis=2)
plot_intensities_veil /= np.max(plot_intensities_veil)

plot_intensities_gray = np.sum(np.dstack(raws), axis=2)
plot_intensities_gray /= np.max(plot_intensities_gray)

# gamma = 2  # Gamma correction
# plot_intensities = plot_intensities ** (1/gamma)
# fig, ax = ip.general_plot()
# ax.plot([0,1],[0,1])
# x = np.linspace(0,1,100)
# ax.plot(x, x**(1/gamma))
# ax.set_aspect('equal')

ul_veil = 0.1  # white limit
plot_intensities_veil[plot_intensities_veil > ul_veil] = ul_veil
plot_intensities_veil /= ul_veil

ul_gray = 0.75  # white limit
plot_intensities_gray[plot_intensities_gray > ul_gray] = ul_gray
plot_intensities_gray /= ul_gray

# %% codecell
# Project clusters back onto image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
tab20 = plt.get_cmap('tab20').colors
# tab20_sort = tab20
# tab_sort = [tab20[i] for i in np.arange(1,20,2).tolist() + np.arange(0,20,2).tolist()]
tab_sort = [tab20[i] for i in np.arange(0,20,2).tolist() + np.arange(1,20,2).tolist()]
grey = tab_sort.pop(7)
# tab_sort = [grey] + tab_sort
labs = np.unique(classifs, return_counts=True)
labs
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_dict = {l:c for l, c in zip(labs_sort, tab_sort)}
im_veil = np.zeros(mask.shape + (len(tab20[0]),))
# im_clust = np.zeros(max.shape + (len(eval(barcode_color.color.values[0])),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    if lab == 'Veillonella':
        col = col_dict[lab]
        im_veil[x,y,:] = np.array(col) * plot_intensities_veil[x,y]
        # col = np.array(col_dict[lab]) * sum_norm[x,y]
    else:
        col = (1,1,1)
        im_veil[x,y,:] = np.array(col) * plot_intensities_gray[x,y]


# %% codecell
# Show the a zoom of the rgb vs the new projection
c=[500,500]
d=[750,750]
im_inches=20
# rgb = np.dstack(raws_max_norm_adj)
ip.subplot_square_images([rgb[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], im_veil[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]], (1,2), im_inches=im_inches)


# %% codecell
# Show the rgb vs the new projection
im_inches=20
rgb = np.dstack(raws_max_norm_adj)
ip.subplot_square_images([rgb, im_veil], (1,2), im_inches=im_inches)
# output_dir = config['output_dir'] + '/test_classif/mat_mul'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# out_bn = output_dir + '/' + sn + '_rgb_plus_pixelclassif'
# ip.save_png_pdf(out_bn)
#


# %% codecell
# save the new projection
im_inches = 5
ip.plot_image(im_veil, im_inches=im_inches)
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_pixelclassif_Veillonella'
# ip.save_png_pdf(out_bn)

# # %% codecell
# col_ordered = [col_dict[l] for l in labs_sort]
# fig, ax = ip.taxon_legend(
#         taxon_names=labs_sort,
#         taxon_colors=col_ordered
#         )
# output_dir = config['output_dir'] + '/test_classif/mat_mul'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# out_bn = output_dir + '/' + sn + '_pixelclassif_full_legend'
# ip.save_png_pdf(out_bn)

# %% md

# ==============================================================================
# ## Overlay veillonella  with MeGAFISH
# ==============================================================================

# %% codecell
# resize hiprfish images
dims = lrg.shape
ul_corner = shp_dff
hipr_veil_resize = resize_hipr(
        im_veil, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )


# %% codecell
# Plot spot overlay on classif
im_inches=10
ceil=1
size_min=0.1
marker_size=100
marker='.'
spot_col=(1,0,1)
edge_col='none'
linewidths=1.5

spot_props_shift = sss_props
fig, ax, cbar = ip.plot_image(
        hipr_veil_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
ref_pts = [c for c in spot_props_shift['centroid'].values]
ref_pts = np.rint(ref_pts).astype(np.int64)
ref_pts_arr = np.array(ref_pts)
# # Plot size as spot intensity
# spot_int = spot_props_shift.max_intensity.values.copy()
# spot_int -= np.min(spot_int)
# spot_int /= np.max(spot_int)
# spot_int[spot_int > ceil] = ceil
# spot_int /= ceil
# spot_int[spot_int < size_min] = size_min
# marker_size_arr = marker_size * spot_int
marker_size_arr=marker_size
ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
            marker=marker, s=marker_size_arr, color=spot_col,
            linewidths=linewidths, edgecolors=edge_col
            )
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spot_overlay_dot'
# ip.save_png_pdf(out_bn)


# %% codecell
# show raw spot on top of classif zoom
ul = 0.25
ll = 0.15
c=[1500,2500]
d=[750,750]
im_inches = 20
# zoom_coords=[c[0], c[0]+d[0], c[1], c[1]+d[1]]

hvr_zoom = hipr_veil_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
fig, ax, cbar = ip.plot_image(
        hvr_zoom, scalebar_resolution=mega_res, im_inches=im_inches
        )
spot_raw = raw_shift[0][:,:,1].copy()
spot_raw -= np.min(spot_raw)
spot_raw /= np.max(spot_raw)
spot_raw[spot_raw > ul] = ul
spot_raw[spot_raw < ll] = 0
spot_raw /= ul
spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
spot_raw_overlay[:,:,0] = spot_raw
spot_raw_overlay[:,:,2] = spot_raw
spot_raw_overlay[:,:,3] = spot_raw
sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(sro_zoom)
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay_veilonella_zoom'
# ip.save_png_pdf(out_bn)




# %% codecell
# save raw spot on top of classif
im_inches = 20

fig, ax, cbar = ip.plot_image(
        hipr_veil_resize, scalebar_resolution=mega_res, im_inches=im_inches
        )
ax.imshow(spot_raw_overlay)
# out_dir = config['output_dir'] + '/pixel_classif_test'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay'
# ip.save_png_pdf(out_bn)



# %% md

# ==============================================================================
# ## Multicolor visualization of spot coloc
# ==============================================================================

# %% codecell
# %% codecell
# adjust intensities
plot_intensities = hipr_max_resize.copy()
plot_intensities /= np.max(plot_intensities)


# # %% codecell
# # Get resized classifs list for pixel inds
# barcodes_int = [int(bc) for bc in barcodes]
# dict_sn_bc = dict(zip(sci_names, barcodes_int))
# im_bc = np.zeros(max_sub.shape)
# classifs_bc = [dict_sn_bc[sn] for sn in classifs]
# pix_ind = np.argwhere(mask)
# for bc, i in zip(classifs_bc, pix_ind):
#     x, y = i
#     # col = col_dict[lab]
#     # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
#     im_bc[x,y] = bc
# hipr_bc_resize = resize_hipr(
#         im_bc, hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
#         )

col_dict['Veillonella']

# %% codecell
# Get veillonella rgb
ulv = 0.25
llv = 0
plot_intensities_veil = (np.clip(plot_intensities, llv, ulv) - llv) / (ulv - llv)
mask_veil = (hipr_bc_resize == 100)
zeros = np.zeros(hipr_bc_resize.shape)
ones = np.ones(hipr_bc_resize.shape)
plot_intensities_veil_mask = plot_intensities_veil * mask_veil
rgb_veil = np.dstack([
        plot_intensities_veil_mask * 1,
        plot_intensities_veil_mask * 0.7,
        plot_intensities_veil_mask * 0.5,
        plot_intensities_veil_mask
        ])
ip.plot_image(np.clip(rgb_veil[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], 0, 1))

# %% codecell
# Get spot RGB
spot_raw_coloc = spot_raw * mask_veil
rgb_spot_coloc = np.dstack([
        spot_raw_coloc * 1,
        spot_raw_coloc * 0,
        spot_raw_coloc * 1,
        spot_raw_coloc
        ])

ip.plot_image(np.clip(rgb_spot_coloc[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], 0, 1))

# %% codecell
# Get spot RGB
spot_raw_noncoloc = spot_raw * ~mask_veil
rgb_spot_noncoloc = np.dstack([
        spot_raw_noncoloc * 1,
        spot_raw_noncoloc * 1,
        spot_raw_noncoloc * 0,
        spot_raw_noncoloc
        ])

ip.plot_image(np.clip(rgb_spot_noncoloc[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], 0, 1))

# # %% codecell
# # Overlay
# rgb_spot_veil = np.clip(rgb_spot + rgb_veil, 0,1)
# ip.plot_image(rgb_spot_veil, im_inches=20)


# %% codecell
im_inches=20
ulg=0.5
llg=0.01

plot_intensities_gray = (np.clip(plot_intensities, llg, ulg) - llg) / (ulg - llg)
fig, ax, cbar = ip.plot_image(plot_intensities_gray, cmap='gray', im_inches=im_inches, clims=(0,3))
ax.imshow(rgb_veil)
ax.imshow(rgb_spot_coloc)
ax.imshow(rgb_spot_noncoloc)


# %% codecell
im_inches=5
pig_zoom = plot_intensities_gray[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
zeros_zoom = zeros[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
rgb_veil_zoom = rgb_veil[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
rgb_spot_coloc_zoom = rgb_spot_coloc[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
rgb_spot_noncoloc_zoom = rgb_spot_noncoloc[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]

fig, ax, cbar = ip.plot_image(pig_zoom, cmap='gray', im_inches=im_inches, clims=(0,2))
# fig, ax, cbar = ip.plot_image(zeros_zoom, cmap='gray',im_inches=im_inches, clims=(0,2))
ax.imshow(rgb_veil_zoom)
ax.imshow(rgb_spot_coloc_zoom)
ax.imshow(rgb_spot_noncoloc_zoom)
out_dir = config['output_dir'] + '/pixel_classif_test'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay_veilonella_zoom'
plt.figure(fig)
# ip.save_png_pdf(out_bn, bbox_inches=False)





# %% md

# ==============================================================================
# ## Project plasmid density onto plaque
# ==============================================================================

# %% codecell
# Get cell

cell = hipr_max_resize
cell_pix_values = np.sort(np.ravel(cell))

# %% codecell
# PLot intensities
thresh_cell=0.05
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
r_um = 5
r_pix = int(r_um / config_mega['resolution'])

#
#
#
area_circle = np.pi * r_um**2
# # Get circle
# def get_circle_mask(dimx, dimy, center, radius):
#     Y, X = np.ogrid[:dimx, :dimy]
#     distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
#     return (distance_from_center <= radius)
# d = r_pix*2 + 1
# circle_mask = get_circle_mask(d, d, (r_pix, r_pix), r_pix)
# area_circle_pix = np.sum(circle_mask)
# ip.plot_image(circle_mask)
#

# # %% codecell
# # Get edge of mask
# cell_mask_bound = ip.find_boundaries(cell_mask, mode='inner')
# # Convolve with circle
# cell_mask_edge = fftconvolve(cell_mask_bound, circle_mask)
# # multiply by mask
# cell_mask_edge *= np.pad(cell_mask, np.int(r_pix), mode='edge')
# cell_mask_edge =
#
# # plot
# ip.plot_image(cell_mask_edge > 0)


# # %% codecell
# # For each pixel coordinate
# thresh_area_frac = 0.9
#
# thresh_circle_area = area_circle_pix * thresh_area_frac
# cell_mask_pad = np.pad(cell_mask, np.int(r_pix), mode='edge')
# area_list = []
# um_sq_per_pix = config['resolution']**2
# for i,j in tqdm(pix_coords):
#     # Get bbox
#     i += r_pix
#     j += r_pix
#     bbox = cell_mask_pad[i-r_pix: i+r_pix+1, j-r_pix: j+r_pix+1]
#     if np.sum(bbox) < thresh_circle_area:
#         # multiply bbox by circle
#         reduced_mask = circle_mask * bbox
#         area = np.sum(reduced_mask) * um_sq_per_pix
#     else:
#         area = area_circle
#     # add to area list
#     area_list.append(area)

# # %% codecell
# # plot edge correction
# cell_edge_map = np.zeros(cell.shape)
# cell_edge_map[pix_inds] = area_list
# fig, ax, cbar = ip.plot_image(cell_edge_map)
# background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
# background = np.dstack(background)
# ax.imshow(background)

# %% codecell
# Calculate spot densities at each cell

# Generate knn object with radius
nbrs = NearestNeighbors(radius=r_pix).fit(spot_coords)
# Get radius knn for each pixel
cell_centroids = hipr_reseg_props.loc[:, 'centroid'].values
cell_coords = [list(c) for c in cell_centroids]
dists, inds = nbrs.radius_neighbors(cell_coords)
# Apply density value to cell pixels
cell_counts = [i.shape[0] for i in inds]
cell_spot_density = np.array(cell_counts) / area_circle


# %% codecell
# Plot
im_inches=10
dict_label_bbox = dict(hipr_reseg_props[['label','bbox']].values)
dict_label_density = dict(zip(hipr_reseg_props.label.values, cell_spot_density))
cell_seg_spot_density = fsi.recolor_image(
        im=hipr_reseg,
        dict_label_bbox=dict_label_bbox,
        dict_label_alt=dict_label_density
        )
cell_seg_spot_density.shape
fig, ax, cbar = ip.plot_image(
        cell_seg_spot_density,
        cmap='cividis',
        scalebar_resolution=config_mega['resolution'],
        im_inches=im_inches
        )
background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
background = np.dstack(background)
ax.imshow(background)

plt.figure(fig)
out_dir = config['output_dir'] + '/density_maps'
if not os.path.exists(out_dir): os.makedirs(out_dir)
output_bn = out_dir + '/' + sn + '_spot_density_map'
# ip.save_png_pdf(output_bn)
plt.figure(cbar)
out_dir = config['output_dir'] + '/density_maps'
if not os.path.exists(out_dir): os.makedirs(out_dir)
output_bn = out_dir + '/' + sn + '_spot_density_map_chan_cbar'
# ip.save_png_pdf(output_bn)
plt.show()
plt.close()

# %% codecell
# Threshold high density areas
thresh_density = 0.1
bool_density = cell_seg_spot_density > thresh_density
cell_seg_spot_density_thresh = cell_seg_spot_density*bool_density
fig, ax, cbar = ip.plot_image(
        bool_density,
        cmap='gray',
        scalebar_resolution=config_mega['resolution']
        )
plt.figure(fig)
out_dir = config['output_dir'] + '/density_maps'
if not os.path.exists(out_dir): os.makedirs(out_dir)
output_bn = out_dir + '/' + sn + '_spot_high_density_group'
# ip.save_png_pdf(output_bn)


# %% codecell
# Get composition in high vs low densiy areas
bool_density_h = np.array(cell_spot_density) > thresh_density
bcs_high_density = hipr_reseg_props.loc[
        bool_density_h,
        'max_intensity'
        ].value_counts().sort_index()
bool_density_l = np.array(cell_spot_density) <= thresh_density
bcs_low_density = hipr_reseg_props.loc[
        bool_density_l,
        'max_intensity'
        ].value_counts().sort_index()
print(bcs_high_density)
print(bcs_low_density)

# %% codecell
# plot composition differences
dims=[3,3]
ft=12
line_col = 'k'

barcodes_order = np.unique([bcs_low_density.index, bcs_high_density.index]).astype(int)
fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
# Noramalize taxon counts within regions
bcs_high_density_norm = bcs_high_density / bcs_high_density.sum()
bcs_low_density_norm = bcs_low_density / bcs_low_density.sum()
# Create an ordered matrix with abundance values
dict_bcs_hdn = bcs_high_density_norm.to_dict()
dict_bcs_ldn = bcs_low_density_norm.to_dict()
bcs_densitygroup = np.empty((len(barcodes_order), 2))
for i, bc in enumerate(barcodes_order):
    try:
        hdn = dict_bcs_hdn[bc]
    except:
        hdn = 0
    try:
        ldn = dict_bcs_ldn[bc]
    except:
        ldn = 0
    bcs_densitygroup[i,:] = [hdn, ldn]

# Plot stacked barplot
for i, bc in enumerate(barcodes_order):
    bottom = np.zeros(2)
    # if i > 0:
    j = i - 1
    while j >= 0:
        bottom += bcs_densitygroup[j,:]
        j -= 1
    if bc == 0:
        color=(0.5,0.5,0.5)
        sci_name='No Barcode'
    else:
        sci_name = dict_bc_sciname[bc]
        color = col_dict[sci_name]
    ax.bar([0,1], bcs_densitygroup[i,:], bottom=bottom, color=color, label=sci_name)
ax.set_xticks(ticks=[0,1], labels=['high','low'])
out_dir = config['output_dir'] + '/density_maps'
if not os.path.exists(out_dir): os.makedirs(out_dir)
output_bn = out_dir + '/' + sn + '_spot_density_group_composition'
# ip.save_png_pdf(output_bn)




# %% md

# ==============================================================================
# ## Look at spectra near spots 
# ==============================================================================

# %% codecell
# get reference spectra
ref_dir = config['hipr_dir'] + '/' + config_hipr['hipr_ref_dir']
fmt = config_hipr['ref_files_fmt']
probe_design_dir = config['hipr_dir'] + '/' + config_hipr['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config_hipr['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
barcodes = probe_design['code'].unique()
barcodes_str = [str(bc).zfill(7) for bc in barcodes]
barcodes_10bit = [bc[0] + '0' + bc[1:4] + '00' + bc[4:] for bc in barcodes_str]
barcodes_b10 = [int(str(bc),2) for bc in barcodes_10bit]
sci_names = [probe_design.loc[probe_design['code'] == bc,'sci_name'].unique()[0]
        for bc in barcodes]
st = config_hipr['ref_chan_start'] + config_hipr['chan_start']
en = config_hipr['ref_chan_start'] + config_hipr['chan_end']
ref_avgint_cols = [i for i in range(st,en)]

ref_spec = []
for bc in barcodes_b10:
    fn = ref_dir + '/'+ fmt.format(bc)
    ref = pd.read_csv(fn, header=None)
    ref = ref[ref_avgint_cols].values
    ref_spec.append(ref)
    # ref_norm = ref / np.max(ref, axis=1)[:,None]
    # weights.append(np.mean(ref_norm, axis=0))

# # max normalized reference
ref_norm = [r / np.max(r, axis=1)[:,None] for r in ref_spec]
weights_max_norm = [np.mean(r, axis=0) for r in ref_norm]

for w, n in zip(weights_max_norm, sci_names):
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
    ax.set_title(n)

# %% codecell
# get raw MGE image 
im_inches = 10

ul = 0.4
ll = 0.1


spot_raw = raw_shift[0][:,:,1].copy()
spot_raw -= np.min(spot_raw)
spot_raw /= np.max(spot_raw)
spot_raw[spot_raw > ul] = ul
spot_raw[spot_raw < ll] = 0
spot_raw /= ul
spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
spot_raw_overlay[:,:,0] = spot_raw
spot_raw_overlay[:,:,2] = spot_raw
spot_raw_overlay[:,:,3] = spot_raw

# %% codecell
# Get full region 
c = [700,500]
d = [2800,3200]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
##############################
# Get zoom region 
c = [1550,1550]
d = [25,30]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
# ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])


# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})


# %% codecell
##############################
# Get zoom region 
c = [1780,1640]
d = [50,50]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
##############################
# Get zoom region 
c = [1420,2850]
d = [50,50]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
# ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})


# %% codecell
##############################
# Get zoom region 
c = [1600,2800]
d = [50,50]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
# ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
##############################
# Get zoom region 
c = [1700,1975]
d = [125,125]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})


# %% codecell
##############################
# Get zoom region 
c = [1500,1600]
d = [30,50]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
##############################
# Get zoom region 
c = [1500,2020]
d = [100,100]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})


# %% codecell
##############################
# Get zoom region 
c = [2833,2050]
d = [15,25]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})


# %% codecell
##############################
# Get zoom region 
c = [3025,2525]
d = [30,30]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})


# %% codecell
##############################
# Get zoom region 
c = [2775,2777]
d = [20,20]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
# ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
##############################
# Get zoom region 
c = [2745,2780]
d = [20,20]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
# ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
##############################
# Get zoom region 
c = [1850,2800]
d = [50,125]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
##############################
# Get zoom region 
c = [1940,2860]
d = [50,50]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:], scalebar_resolution=mega_res, im_inches=im_inches
        )

# sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])

# %% codecell
# Get spectra 

stack_resize_zoom = hipr_stacksmooth_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
spectra_zoom = stack_resize_zoom[np.ones(stack_resize_zoom.shape[:2], dtype=bool)]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})


# %% codecell
##############################
# Plot the 561 channel vs the 633 channel


c = [500,500]
d = [3000,3200]

fig, ax, cbar = ip.plot_image(
        hipr_rgb_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],2], 
        scalebar_resolution=mega_res, 
        im_inches=im_inches,
        cmap='gray'
        )

ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])


# %% codecell
##############################
# Plot the ROX signal vs the 633 channel


c = [500,500]
d = [3000,3200]

fig, ax, cbar = ip.plot_image(
        np.sum(hipr_stack_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1],48:], axis=2), 
        scalebar_resolution=mega_res, 
        im_inches=im_inches,
        cmap='gray',
        clims=(0,2.5)
        )

ax.imshow(spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:])




# %% codecell
# im_inches=20
# ceil=1
# size_min=0.1
# marker_size=100
# marker='.'
# spot_col=(1,0,1)
# edge_col='none'
# linewidths=1.5

# spot_props_shift = sss_props
# fig, ax, cbar = ip.plot_image(
#         hipr_rgb_resize, scalebar_resolution=mega_res, im_inches=im_inches
#         )
# ref_pts = [c for c in spot_props_shift['centroid'].values]
# ref_pts = np.rint(ref_pts).astype(np.int64)
# ref_pts_arr = np.array(ref_pts)
# # # Plot size as spot intensity
# # spot_int = spot_props_shift.max_intensity.values.copy()
# # spot_int -= np.min(spot_int)
# # spot_int /= np.max(spot_int)
# # spot_int[spot_int > ceil] = ceil
# # spot_int /= ceil
# # spot_int[spot_int < size_min] = size_min
# # marker_size_arr = marker_size * spot_int
# marker_size_arr=marker_size
# ax.scatter(ref_pts_arr[:,1], ref_pts_arr[:,0],
#             marker=marker, s=marker_size_arr, color=spot_col,
#             linewidths=linewidths, edgecolors=edge_col
#             )
# ax.set_xlim(c[1], c[1]+d[1])
# ax.set_ylim(c[0], c[0]+d[0])