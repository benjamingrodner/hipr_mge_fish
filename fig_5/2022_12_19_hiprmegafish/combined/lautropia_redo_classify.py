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
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2022_12_19_hiprmegafish/combined'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory


# %% md

# Load all the variables from the segmentation pipeline

# %% codecell
config_fn = 'config.yaml' # relative path to config file from workdir

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
sn_i = 11
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
raw_regex = data_dir + '/' + sn + '_mode_[0-9][0-9][0-9].czi'
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
ip.plot_image(np.dstack(raws_max_norm_adj), im_inches=im_inches)
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_rgb'
ip.save_png_pdf(out_bn)
plt.show()
plt.close()



# %% codecell
# Register images
shift_vectors = fsi._get_shift_vectors(raws_max)
print(shift_vectors)
# %% codecell
raws_shift = fsi._shift_images(raws, shift_vectors, max_shift=500)
print(raws_shift[0].shape)

# %% codecell
# show unshifted rgb overlay on zoom
c = [1000, 2000]  # corner
# c = [1400, 1400]  # corner
w = [1000,1000]  # height and width
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

raws_shift_max_norm_adj_sub = np.dstack(raws_shift_max_norm_adj)[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
ip.plot_image(raws_shift_max_norm_adj_sub, im_inches=im_inches)
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape//(im_inches*2))
plt.show()
plt.close()


# %% codecell
stack = np.dstack(raws_shift)
stack_zoom = stack[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
stack.shape


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
ip.save_png_pdf(out_bn)

# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )
output_dir = config['output_dir'] + '/test_classif/mat_mul'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_classif_col_legend'
ip.save_png_pdf(out_bn)



# %% codecell
# show rgb overlay on zoom
c = [2000, 0]  # corner
# c = [1400, 1400]  # corner
w = [1000,1000]  # height and width
# ulims = (1,1,0.45)

im_inches=10
stack_zoom = np.dstack(raws_max_norm_adj)[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
im_classif_rgb_zoom = im_classif_rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
ip.subplot_square_images([stack_zoom, im_classif_rgb_zoom], (1,2), im_inches=im_inches)
# ip.plot_image(stack_zoom, im_inches=im_inches)
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape//(im_inches*2))
# ip.plot_image(im_classif_rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:], im_inches=im_inches)


# %% md

# ==============================================================================
# ## Overlay with MeGAFISH
# ==============================================================================

### Get Shift vectors for mega images

# %% codecell
# Get the Rescaled hiprfish image
from cv2 import resize, INTER_CUBIC, INTER_NEAREST



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
    if out_fn: np.save(out_fn, hipr_resize)
    return hipr_resize

mega_res = config_mega['resolution']
hipr_res = config_hipr['resolution']
hipr_max_resize = resize_hipr(
        np.max(stack, axis=2), hipr_res, mega_res
        )

# %% codecell
# Get mega cell image

mega_out_dir = config['mega_dir'] + '/' + config_mega['output_dir']

mega_raw_fn = mega_out_dir + '/' + config_mega['raw_fmt'].format(sample_name=sn)
mega_raw = np.load(mega_raw_fn)
reg_ch = config['mega_reg_ch_index']
mega_cell = mega_raw[:,:,config_mega['cell_seg']['channels'][reg_ch]]

# mega_data_dir = config['mega_dir'] + '/' + config_mega['input_dir']
# mega_raw_fn = mega_data_dir + '/' + sn + config_mega['input_ext']
# mega_raw = bioformats.load_image(mega_raw_fn)

mega_raw.shape


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
mega_cs_fn = mega_out_dir + '/' + config_mega['cell_seg_area_filt_fmt'].format(sample_name=sn,cell_chan=c_ch)
mega_ss_fn = mega_out_dir + '/' + config_mega['spot_seg_cid_filt_fmt'].format(sample_name=sn, cell_chan=c_ch, spot_chan=s_ch)
mega_cs = np.load(mega_cs_fn)
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
cs_shifts = shift_mega(mega_cs)
ss_shifts = shift_mega(mega_ss)
# Run the props function
cs_shifts[0].shape
css_props = get_mega_props(cs_shifts[0], raw_shift[0], ch_=c_ch)
sss_props = get_mega_props(ss_shifts[0], raw_shift[0], ch_=s_ch)

# %% codecell
# resize hiprfish images
hipr_classif_resize = resize_hipr(
        im_classif_rgb, hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )
hipr_maz_resize = resize_hipr(
        np.max(stack, axis=2), hipr_res, mega_res, dims=dims, ul_corner=ul_corner
        )
# %% codecell
# plot cell overlay
im_inches=10

m_ul=0.25
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
c=[0,2000]
d=[2000,2000]
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
sigma=3

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

# %% codecell
# Cluster
nclust=8

kmeans = KMeans(n_clusters=nclust, random_state=42).fit_predict(las_max_sub_pix)

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
im_inches=10
ip.subplot_square_images([stack_zoom, im_clust], (1,2), im_inches=im_inches)

# %% codecell
n = 100
labs = np.unique(kmeans, return_counts=True)
len(kmeans)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
spec_pix_sub = np.dstack(raws_sub_smooth)[mask_sub]
spec_pix_sub.shape
for l in labs_sort[:20]:
    bool = (kmeans == l)
    # spec_ = spec_pix_scaled[bool,:]
    # spec_ = spec_pix_norm[bool,:]
    spec_ = spec_pix_sub[bool,:]
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
