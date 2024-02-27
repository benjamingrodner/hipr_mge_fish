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
from time import time

# %% md

# Move to the workdir

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2023_02_09_hiprmega/HiPRFISH'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory


# %% md

# Load all the variables from the segmentation pipeline

# %% codecell
config_fn = 'config_hipr.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%load_ext autoreload
%autoreload 2

sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import fn_general_use as fgu
import image_plots as ip
import segmentation_func as sf
import fn_hiprfish_classifier as fhc
import fn_spectral_images as fsi

# %% md

# ==============================================================================
# ## Cluster spectra
# ==============================================================================

# %% codecell
# Pick an image
input_table = pd.read_csv(config['images']['image_list_table'])
input_table

# %% codecell
sn_i = 11
sn = input_table['IMAGES'][sn_i]
sn

# %% codecell
# Load spectra

props_fn = config['output_dir'] + '/' + config['seg_props_fmt'].format(sample_name=sn)
props = pd.read_csv(props_fn)
avgint_cols = [str(i) for i in range(config['chan_start'],config['chan_end'])]
cell_spec_full = props[avgint_cols].values
cell_spec_full.shape

# %% codecell
# Normalize spectra
csf_norm = cell_spec_full / np.max(cell_spec_full, axis=1)[:,None]
# csf_scaler = StandardScaler().fit(csf_norm)
# csf_scale = csf_scaler.transform(csf_norm)


# %% codecell
# Build distance matrix for spectra based on ecs metric
spec_arr = cell_spec_full

shp = spec_arr.shape[0]
spec_dmat = np.zeros((shp,shp))
for i in tqdm(range(shp)):
    for j in range(shp):
        si = spec_arr[i,:]
        if not i == j:
            sj = spec_arr[j,:]
            spec_dmat[i,j] = fhc.euclid_dist_cumul_spec(si, sj)
csf_dmat_ecsn = spec_dmat
# %% codecell
bins=100
_ = plt.hist(np.ravel(csf_dmat_ecsn), bins=bins)


# %% codecell
# Run agglomerative clustering
distance_threshold = 0.2
n_clusters=None

aggl_obj = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        affinity='precomputed',
        linkage='average'
        )
labels_aggl_ecsn = aggl_obj.fit_predict(csf_dmat_ecsn)
np.unique(labels_aggl_ecsn).shape


# %% codecell
# Plot the spectra
classif_out_dir = config['output_dir'] + '/ecs_agglomerative_cell_spectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_aggl_ecsn, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (labels_aggl_ecsn == l)
    spec_ = cell_spec_full[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/spectra_agglomerative_raw_cluster_' + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()


# %% md

# ==============================================================================
# ## Pixel clustering
# ==============================================================================

# %% codecell
# Load the registered image
ims = []
for fmt in ['sum_fmt','seg_fmt','reg_fmt']:
    fn = config['output_dir'] + '/' + config[fmt].format(sample_name=sn)
    ims.append(np.load(fn))

sum_im, seg, reg = ims

# %% codecell
# get a subset
i = [1500,1500]
w = [100,100]
clims=(0,'max')

sum_sub = sum_im[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
ip.plot_image(sum_sub, cmap='inferno', clims=clims)

# %% codecell
# subset all images
seg_sub = seg[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
reg_sub = reg[i[0]:i[0]+w[0], i[1]:i[1]+w[1], :]
ip.plot_image(seg_sub)

# %% codecell
# smoothing
sigma=1.4
reg_sub_smooth = np.empty(reg_sub.shape)
for i in range(reg_sub.shape[2]):
    reg_sub_smooth[:,:,i] = gaussian_filter(reg_sub[:,:,i], sigma=sigma)

# %% codecell
# get spectra and normalize
mask_sub = seg_sub > 0
pix_spec = reg_sub_smooth[mask_sub, :]
ps_norm = pix_spec / np.max(pix_spec, axis=1)[:,None]

# %% codecell
# Build distance matrix for spectra based on ecs metric
spec_arr = pix_spec

shp = spec_arr.shape[0]
spec_dmat = np.zeros((shp,shp))
for i in tqdm(range(shp)):
    for j in range(shp):
        si = spec_arr[i,:]
        if not i == j:
            sj = spec_arr[j,:]
            spec_dmat[i,j] = fhc.euclid_dist_cumul_spec(si, sj)

# %% codecell
bins=100
_ = plt.hist(np.ravel(spec_dmat), bins=bins)


# %% codecell
# Run agglomerative clustering
distance_threshold = 1.0
n_clusters = None

aggl_obj = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=n_clusters,
        affinity='precomputed',
        linkage='average'
        )
labels_aggl_ecsn = aggl_obj.fit_predict(spec_dmat)
np.unique(labels_aggl_ecsn).shape

# %% codecell
# spectra plots
# Plot the spectra
plot_arr = pix_spec

classif_out_dir = config['output_dir'] + '/ecs_agglomerative_cell_spectra/pixel_spectra_subset'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(labels_aggl_ecsn, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (labels_aggl_ecsn == l)
    spec_ = plot_arr[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/pix_spectra_agglomerative_raw_cluster_' + str(l).zfill(2)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# Re project the pixels onto the image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask_sub)
# for each index draw the color
tab10 = plt.get_cmap('tab10').colors
col_dict = {l:c for l, c in zip(labs_sort,tab10)}

im_clust = np.zeros(seg_sub.shape + (len(tab10[0]),))
for lab, ind in zip(labels_aggl_ecsn, pix_ind):
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
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )

# %% codecell
# Plot cell spectra
props_sub = props[props['label'].isin(np.unique(seg_sub))]
cell_spec_sub = props_sub[avgint_cols].values
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, cell_spec_sub, {'lw':0.5,'alpha':0.5,'color':'r'})

# %% md

# ==============================================================================
# ## Nearest neighbor classifier
# ==============================================================================

# %% codecell
# Load the reference spectra
ref_dir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2023_02_16_ecreference/HiPRFISH/seg_props'
fmt = '2023_02_16_ecreference_code_{}_fov_01_seg_props.csv'
barcodes = ['00001','00010','00100','01000','10000']
ref_data = pd.DataFrame([])
for bc in barcodes:
    fn = ref_dir + '/'+ fmt.format(bc)
    df = pd.read_csv(fn)
    df['barcode'] = bc
    ref_data = ref_data.append(df)
ref_spec = ref_data[avgint_cols].values
ref_data_codes = ref_data['barcode'].values
ref_spec_norm = ref_spec / np.max(ref_spec, axis=1)[:,None]

# %% codecell
# Train the knn
training = ref_spec_norm
target = csf_norm

# neigh_obj = NearestNeighbors(n_neighbors=1)
neigh_obj = NearestNeighbors(n_neighbors=1, metric=fhc.euclid_dist_cumul_spec)
neigh_fit = neigh_obj.fit(training)
# classify the spectra
dist, ind = neigh_fit.kneighbors(target)

# %% codecell
# Filter based on distance
thresh_dist=1
dist_sort = np.sort(dist, axis=0)
fig, ax = ip.general_plot()
ax.plot(np.arange(dist.shape[0]), dist_sort)
ax.plot([0,dist.shape[0]],[thresh_dist]*2,'k')


# %% codecell
labs = np.unique(classifs)
classifs = ref_data_codes[ind]
# classifs[dist>thresh_dist] = 'none'
classifs = classifs.squeeze()

# %% codecell
# Plot the spectra groups
plot_arr = cell_spec_full

classif_out_dir = config['output_dir'] + '/nnspectra'
if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    spec_ = plot_arr[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('Barcode: ' + str(l).zfill(2))
    out_bn = classif_out_dir + '/nn_spectra_{}'.format(sn) + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# label to cluster  dictionairy
# classifs_float = [float(re.sub('_error','.404', str(bc[0]))) for bc in classifs]
dict_label_clust = dict(zip(props['label'], classifs))
# cluster to color dict
clusters = np.unique(classifs, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab10').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
# i_none = np.argwhere(np.array(clusters_sort) == 'none')[0][0]
# colors[i_none] = (0.5,0.5,0.5)
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props['label'], props['bbox']))
# re color image
knn_col = fsi.recolor_image(seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(knn_col, scalebar_resolution=config['resolution'])
# out_bn = classif_out_dir + '/{}_nn_classif_ecs_col'.format(sn)
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot the legend
probe_design_dir = config['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
taxon_names = [dict_bc_tax[int(bc)] if not bc == 'none' else bc for bc in clusters_sort[:20]]
col_ordered = [dict_clust_col[l] for l in clusters_sort]
fig, ax = ip.taxon_legend(
        taxon_names=taxon_names,
        taxon_colors=col_ordered,
        taxon_counts=sorted(clusters[1], reverse=True)
        )

# %% codecell
# write to the pipeline file
props_out_fn = config['output_dir'] + '/' + config['props_classif_fmt'].format(sample_name=sn)

props.columns
props['cell_barcode'] = classifs
props['nn_dist'] = dist
props.to_csv(props_out_fn)


# %% md

# ==============================================================================
# ## Intensity filter
# ==============================================================================

# %% codecell
# Pick max threshold
thresh_max=0.075
maxs = np.max(cell_spec_full, axis=1)
maxs_sort = np.sort(maxs)
fig, ax = ip.general_plot()
ax.plot(np.arange(maxs_sort.shape[0]), maxs_sort)
ax.plot([0,dist.shape[0]],[thresh_max]*2,'k')


# %% codecell
# Filter low intensity
classifs[maxs<thresh_max] = 'none'
classifs = classifs.squeeze()

# %% codecell
# Plot the spectra groups
plot_arr = cell_spec_full

# classif_out_dir = config['output_dir'] + '/'
# if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    spec_ = plot_arr[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('Barcode: ' + str(l).zfill(2))
    # out_bn = classif_out_dir + '/pix_spectra_agglomerative_raw_cluster_' + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# label to cluster  dictionairy
# classifs_float = [float(re.sub('_error','.404', str(bc[0]))) for bc in classifs]
dict_label_clust = dict(zip(props['label'], classifs))
# cluster to color dict
clusters = np.unique(classifs, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab10').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
i_none = np.argwhere(np.array(clusters_sort) == 'none')[0][0]
colors[i_none] = (0.5,0.5,0.5)
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props['label'], props['bbox']))
# re color image
knn_col = fsi.recolor_image(seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(knn_col, scalebar_resolution=config['resolution'])
# out_bn = classif_out_dir + '/{}_nn_classif_ecs_col'.format(sn)
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot the legend
probe_design_dir = config['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
taxon_names = [dict_bc_tax[int(bc)] if not bc == 'none' else bc for bc in clusters_sort[:20]]
col_ordered = [dict_clust_col[l] for l in clusters_sort]
fig, ax = ip.taxon_legend(
        taxon_names=taxon_names,
        taxon_colors=col_ordered,
        taxon_counts=sorted(clusters[1], reverse=True)
        )

# %% md

# ==============================================================================
# ## laser norm classifier
# ==============================================================================

# %% codecell
# Norm reference by laser
chan_inds = [0,23,43,57]
ref_las_specs = []
for i in range(len(chan_inds) - 1):
    c0, c1 = chan_inds[i], chan_inds[i+1]
    spec = ref_spec[:,c0:c1]
    spec_norm = (spec + 1e-5) / (np.max(spec, axis=1)[:,None] + 1e-5)
    ref_las_specs.append(spec_norm)


# %% codecell
#  knn trained for each laser
knns = []
for rls in ref_las_specs:
    neigh_obj = NearestNeighbors(n_neighbors=1, metric=fhc.euclid_dist_cumul_spec)
    knns.append(neigh_obj.fit(rls))

# %% codecell
# norm spectra by laser
las_specs = []
for i in range(len(chan_inds) - 1):
    c0, c1 = chan_inds[i], chan_inds[i+1]
    spec = cell_spec_full[:,c0:c1]
    spec_norm = (spec + 1e-5) / (np.max(spec, axis=1)[:,None] + 1e-5)
    las_specs.append(spec_norm)

# %% codecell
# Get knn for each laser
dists = []
inds = []
for spec, knn in zip(las_specs, knns):
    dist, ind = knn.kneighbors(spec)
    dists.append(dist)
    inds.append(ind)


# %% codecell
# write barcodes in array
barcodes = np.array(['00001','00010','00100','01000','10000'])
# convert indices into barcode indices
bc_inds = []
for ind in inds:
    bcs = ref_data_codes[ind]
    bc_inds.append([np.argwhere(barcodes == bc)[0][0] for bc in bcs])
# add one to distances
dists_plusone = [dist + 1 for dist in dists]
# create nx5 matrix for each laser
# assign inverse of distance to knn barcode indices in matrices
d_factor = 4
wt_mats = []
for i, d in zip(bc_inds, dists_plusone):
    mat = np.zeros((cell_spec_full.shape[0], len(barcodes)))
    rows = np.arange(mat.shape[0]).astype(int).tolist()
    mat[(rows, i)] = 1 / (d.squeeze()**d_factor)
    wt_mats.append(mat)


# %% codecell
# Get relative weights between lasers for each cell
l_factor=0
las_maxs = [np.max(spec, axis=1)[:,None] for spec in las_specs]
las_maxs_norm = np.hstack(las_maxs)
las_maxs_norm /= np.max(las_maxs_norm, axis=1)[:,None]
rel_mats = [las_maxs_norm[:,i]**l_factor for i in range(las_maxs_norm.shape[1])]

# %% codecell
# Weight each laser by relative weight
wt_mats_rel = [mat*rel[:,None] for mat, rel in zip(wt_mats, rel_mats)]

# %% codecell
# Sum laser matrices
wt_mat_sum = np.zeros(wt_mats_rel[0].shape)
for mat in wt_mats_rel:
    wt_mat_sum += mat


# %% codecell
# Assign max bc weight as to cell
bc_max_inds = np.argmax(wt_mat_sum, axis=1)
label_lasnorm = np.array([barcodes[i] for i in bc_max_inds])
label_lasnorm.shape

# %% codecell
# Plot the spectra groups
plot_arr = cell_spec_full
# classif_out_dir = config['output_dir'] + '/'
# if not os.path.exists(classif_out_dir): os.makedirs(classif_out_dir)
labs = np.unique(label_lasnorm, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (label_lasnorm == l)
    spec_ = plot_arr[bool,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_, {'lw':0.5,'alpha':0.5,'color':'r'})
    ax.set_title('Barcode: ' + str(l).zfill(2))
    # out_bn = classif_out_dir + '/pix_spectra_agglomerative_raw_cluster_' + str(l).zfill(2)
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()

# %% codecell
# label to cluster  dictionairy
# classifs_float = [float(re.sub('_error','.404', str(bc[0]))) for bc in classifs]
dict_label_clust = dict(zip(props['label'], label_lasnorm))
# cluster to color dict
clusters = np.unique(label_lasnorm, return_counts=True)
clusters_sort = [x for _, x in sorted(zip(clusters[1],clusters[0]), reverse=True)]
colors = list(plt.get_cmap('tab10').colors)
if len(colors) > len(clusters_sort):
    colors = colors[:len(clusters_sort)]
elif len(colors) < len(clusters_sort):
    colors += [(0.5,0.5,0.5)]*(len(clusters_sort) - len(colors))
# i_none = np.argwhere(np.array(clusters_sort) == 'none')[0][0]
# colors[i_none] = (0.5,0.5,0.5)
dict_clust_col = dict(zip(clusters_sort, colors))
# Label to color dict
dict_label_col = {l:dict_clust_col[c] for l, c in dict_label_clust.items()}
# label to bbox dictionary
dict_label_bbox = dict(zip(props['label'], props['bbox']))
# re color image
knn_col = fsi.recolor_image(seg, dict_label_bbox, dict_label_col, threeD=3)

# %% codecell
# Plot the recolored image
im_inches = 5
ip.plot_image(knn_col, scalebar_resolution=config['resolution'])
# out_bn = classif_out_dir + '/{}_nn_classif_ecs_col'.format(sn)
# ip.save_png_pdf(out_bn, dpi=np.max(hipr_seg_resize.shape)//im_inches)

# %% codecell
# plot the legend
probe_design_dir = config['__default__']['PROBE_DESIGN_DIR']
probe_design_fn = probe_design_dir + '/' + config['probe_design_filename']
probe_design = pd.read_csv(probe_design_fn)
dict_bc_tax = dict(zip(probe_design['code'],probe_design['sci_name']))
taxon_names = [dict_bc_tax[bc] if not bc == 'none' else bc for bc in clusters_sort[:20]]
col_ordered = [dict_clust_col[l] for l in clusters_sort]
fig, ax = ip.taxon_legend(
        taxon_names=taxon_names,
        taxon_colors=col_ordered,
        taxon_counts=sorted(clusters[1], reverse=True)
        )

# %% md

# ==============================================================================
# ## check 633 channel bleed through to 561
# ==============================================================================

# %% codecell
# Look at 647 reference spectrum
ref_fmt = config['hipr_ref_dir'] + '/' + config['ref_files_fmt']
ref_r7 = pd.read_csv(ref_fmt.format(int('0001000000',2))).values
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, ref_r7, {'lw':0.5,'alpha':0.5,'color':'r'})


# %% md

# Get nearest neighbor for spots

# %% codecell
# get megafish config
mega_dir = '../MeGAFISH'
config_mega_fn = mega_dir + '/config_mgefish.yaml'
with open(config_mega_fn, 'r') as f:
    config_mega = yaml.safe_load(f)

hiprmega_dir = '../HiPR_MeGA'
config_hiprmega_fn = hiprmega_dir + '/config.yaml'
with open(config_hiprmega_fn, 'r') as f:
    config_hiprmega = yaml.safe_load(f)

# %% codecell
# load spot props
spot_props_shift_fmt = (hiprmega_dir + '/' + config_hiprmega['output_dir']
        + '/' + config_hiprmega['mega']['spot_props_shift'])
spot_props_shift = pd.read_csv(spot_props_shift_fmt.format(sample_name=sn, cell_chan=0,spot_chan=1))
spots = [list(eval(c)) for c in spot_props_shift['centroid']]
spot_props_shift.columns

# %% codecell
cell_props_res_fmt = (hiprmega_dir + '/' + config_hiprmega['output_dir']
        + '/' + config_hiprmega['hipr']['props_resize'])
cell_props_res = pd.read_csv(cell_props_res_fmt.format(sample_name=sn))
centroids = [list(eval(c)) for c in cell_props_res['centroid']]

cell_props_res.columns

# %% codecell
# Get nearest neighbors
n_neighbors = 1
nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(centroids)
dist, inds = nbrs.kneighbors(spots, return_distance=True)
labels = cell_props_res.loc[inds.squeeze(), 'label'].values
dist_sort = np.sort(labels.squeeze())
plt.plot(np.arange(dist_sort.shape[0]), dist_sort)
plt.show()
plt.close()
neigh_cells = cell_props_res.loc[inds.squeeze(), 'centroid'].values
fig, ax = ip.general_plot(dims=(10,10))
for s, c in zip(spots, neigh_cells):
    c = eval(c)
    ax.plot([s[0],c[0]],[s[1],c[1]], 'k')

# %% codecell
# Get spectra for nearest neighbors
cell_props_fmt = config['output_dir'] + '/' + config['classif_filt_fmt']
cell_props = pd.read_csv(cell_props_fmt.format(sample_name=sn))
spectra_spot = cell_props.loc[inds.squeeze(), avgint_cols].values
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spectra_spot, {'lw':0.5,'alpha':0.5,'color':'r'})


# %% codecell
# Get random spectra from spots that are not neighbors
cells_rand = cell_props.loc[~cell_props.index.isin(inds.squeeze()), :].sample(len(spots))
spec_rand = cells_rand.loc[:,avgint_cols]
fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.5,'alpha':0.5,'color':'r'})
plt.show()
plt.close()
fig, ax = ip.general_plot(dims=(10,10))
for s, c in zip(spots, neigh_cells):
    c = eval(c)
    ax.plot([s[0],c[0]],[s[1],c[1]], 'k', lw=2)

cells_rand.columns
cells_rand = cell_props_res.loc[cells_rand.index, ['centroid-0','centroid-1']]
ax.scatter(cells_rand['centroid-0'], cells_rand['centroid-1'], c='r')


# %% md

# ==============================================================================
# ## Fix image registration
# ==============================================================================

# %% codecell
# raw images
data_dir = (config['__default__']['DATA_DIR'])
# data_dir = '../../../../data/Combined_taxonomic_mapping_and_MGE_mapping/prophage_termL_host_association_images'
raw_regex = data_dir + '/' + sn + '_mode_[0-9][0-9][0-9]_stitch.czi'

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

print(raws[0].shape)

# %% codecell
# show rgb overlay
ulims = (0.45,0.45,0.45)

im_inches=20
raws_max = [np.max(im, axis=2) for im in raws]
sh_min = [np.min([r.shape[i] for r in raws_max]) for i in [0,1]]
raws_max = [r[:sh_min[0],:sh_min[1]] for r in raws_max]
raws_max_norm = [im / np.max(im) for im in raws_max]
raws_max_norm_adj = []
for im, ul in zip(raws_max_norm, ulims):
    im[im>ul] = ul
    raws_max_norm_adj.append(im/ul)
ip.plot_image(np.dstack(raws_max_norm_adj), im_inches=im_inches)
out_dir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/termL_taxon_mapping/rgbs'
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape[0]//(im_inches*2))
plt.show()
plt.close()


# %% codecell
# show rgb overlay on zoom
c = [0, 0]  # corner
# c = [1400, 1400]  # corner
w = [1000,1000]  # height and width
ulims = (0.45,0.45,0.45)


stack_zoom = np.dstack(raws_max_norm_adj)[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
ip.plot_image(stack_zoom, im_inches=im_inches)
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape//(im_inches*2))
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
# show rgb overlay
ulims = (0.45,0.45,0.45)

im_inches=20
raws_max = [np.max(im, axis=2) for im in raws_shift]
sh_min = [np.min([r.shape[i] for r in raws_max]) for i in [0,1]]
raws_max = [r[:sh_min[0],:sh_min[1]] for r in raws_max]
raws_max_norm = [im / np.max(im) for im in raws_max]
raws_max_norm_adj = []
for im, ul in zip(raws_max_norm, ulims):
    im[im>ul] = ul
    raws_max_norm_adj.append(im/ul)
ip.plot_image(np.dstack(raws_max_norm_adj), im_inches=im_inches)
out_dir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/termL_taxon_mapping/rgbs'
plt.savefig(out_dir + '/' + sn + '_rgb_shift.png', dpi=raws[0].shape[0]//(im_inches*2))
plt.show()
plt.close()
# %% codecell
raws_shift_max = [np.max(im, axis=2) for im in raws_shift]
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


# %% md

# Look at zooms on spectra

# %% codecell
stack = np.dstack(raws_shift)
stack_zoom = stack[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
stack.shape

# %% codecell
chans = [4,33,44,47]
cols = [(0.3,0,0),(0.3,0,0.5),(0,0.5,0.5),(0.3,0.5,0)]
ulims = (1,0.7,0.7,0.7)


raws_chan = [stack[:,:,ch] for ch in chans]
# raws_chan = [stack[:,:,c] for c in chans]
raws_chan_norm = [im / np.max(im) for im in raws_chan]
raws_chan_norm_adj = []
for im, ul in zip(raws_chan_norm, ulims):
    im[im>ul] = ul
    raws_chan_norm_adj.append(im/ul)
raws_chan_col = [im[...,None] * np.array(c)[None,None,:] for c, im in zip(cols, raws_chan_norm_adj)]
chan_rgb = np.zeros(raws_shift_max_norm_adj_sub.shape)
for im in raws_chan_col:
    chan_rgb += im
ip.plot_image(chan_rgb, im_inches=im_inches)

# output_dir = config['output_dir'] + '/test_classif/mat_mul'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# out_bn = output_dir + '/' + sn + '_zoom_rgb'
# ip.save_png_pdf(out_bn)


# %% codecell
# shifted
chans = [14]

# stack = np.dstack(raws_shift)
raws_chan = [stack[:,:,ch] for ch in chans]
# raws_chan = [stack[:,:,c] for c in chans]
raws_chan_norm = [im / np.max(im) for im in raws_chan]
raws_chan_norm_adj = []
for im, ul in zip(raws_chan_norm, ulims):
    im[im>ul] = ul
    raws_chan_norm_adj.append(im/ul)
raws_chan_col = [im[...,None] * np.array(c)[None,None,:] for c, im in zip(cols, raws_chan_norm_adj)]
chan_rgb = np.zeros(raws_shift_max_norm_adj_sub.shape)
for im in raws_chan_col:
    chan_rgb += im
ip.plot_image(chan_rgb, im_inches=im_inches)


# %% md

# ==============================================================================
# ## Standard Pixel classifier
# ==============================================================================

# %% codecell
# smoothing
sigma=4

stack_smooth = np.empty(stack.shape)
for i in range(stack.shape[2]):
    stack_smooth[:,:,i] = gaussian_filter(stack[:,:,i], sigma=sigma)

# %% codecell
# Get maximum intensity mask
max_thresh = 0.05
im_inches=10

max_unsmooth = np.max(stack, axis=2)
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
ip.subplot_square_images([max_unsmooth, mask, max_masked], (1,3), im_inches=im_inches)
# %% codecell
spec_pix = stack_smooth[mask > 0]
spec_pix_norm = spec_pix / np.max(spec_pix, axis=1)[:,None]



# # %% codecell
# # # Load the registered image
# # ims = []
# # for fmt in ['sum_fmt','seg_fmt','reg_fmt']:
# #     fn = config['output_dir'] + '/' + config[fmt].format(sample_name=sn)
# #     ims.append(np.load(fn))
# #
# # sum_im, seg, reg = ims
#
# # %% codecell
# # get a subset
# i = [750,750]
# w = [250,250]
# clims=(0,'max')
#
# sum_sub = sum_im[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
# ip.plot_image(sum_sub, cmap='inferno', clims=clims)
#
# # %% codecell
# # subset all images
# seg_sub = seg[i[0]:i[0]+w[0], i[1]:i[1]+w[1]]
# reg_sub = reg[i[0]:i[0]+w[0], i[1]:i[1]+w[1], :]
# ip.plot_image(seg_sub)

# %% codecell
# Plot spectra

# %% codecell
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(spec_pix_norm)
labels_kmeans.shape[0]

# %% codecell
n = 1000
labs = np.unique(labels_kmeans, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (labels_kmeans == l)
    spec_ = spec_pix_norm[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.1,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))

# %% codecell
# Re project the pixels onto the image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
tab10 = plt.get_cmap('tab10').colors
col_dict = {l:c for l, c in zip(labs_sort, tab10)}
im_clust = np.zeros(max_sub.shape + (len(tab10[0]),))
for lab, i in zip(labels_kmeans, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * max_unsmooth[x,y]

# %% codecell
im_inches=10
ip.plot_image(im_clust, im_inches=im_inches)

# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )

# %% md

# Measure distance on pixels

# %% codecell
# Load the reference spectra
ref_dir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2023_02_16_ecreference/HiPRFISH/seg_props'
fmt = '2023_02_16_ecreference_code_{}_fov_01_seg_props.csv'
barcodes = ['00001','00010','00100','01000','10000']
avgint_cols = [str(i) for i in range(config['chan_start'],config['chan_end'])]
# for bc in barcodes:
# bc = barcodes[0]
# fn = ref_dir + '/'+ fmt.format(bc)
# ref = pd.read_csv(fn)[avgint_cols].values
# ref_norm = ref / np.max(ref_norm, axis=1)[:,None]
# neigh_obj = NearestNeighbors(n_neighbors=1, metric=fhc.euclid_dist_cumul_spec)
# neigh_fit = neigh_obj.fit(ref_norm)
# dist, _ = neigh_fit.kneighbors(spec_pix_norm)
# intensity = 1 / (dist + 1e-10)
# intensity_norm = intensity / np.max(intensity)
# im_bc = np.zeros(max_sub.shape + (3,))
# im_bc[mask] = intensity_norm

# # %% codecell
# im_inches = 10
# color = [1,0,1]
#
# im_bc_col = im_bc * np.array(color)[None, None, :]
# ip.plot_image(im_bc_col, im_inches=im_inches)

ref_data = pd.DataFrame([])
for bc in barcodes:
    fn = ref_dir + '/'+ fmt.format(bc)
    df = pd.read_csv(fn)
    df['barcode'] = bc
    ref_data = ref_data.append(df)
ref_spec = ref_data[avgint_cols].values
ref_data_codes = ref_data['barcode'].values
ref_spec_norm = ref_spec / np.max(ref_spec, axis=1)[:,None]

# %% codecell
# Train the knn
training = ref_spec_norm
target = spec_pix_norm

# neigh_obj = NearestNeighbors(n_neighbors=1)
# neigh_obj = NearestNeighbors(n_neighbors=1, metric=fhc.euclid_dist_cumul_spec)
neigh_obj = NearestNeighbors(n_neighbors=1, metric=fhc.channel_cosine_intensity_5b_v2)

neigh_fit = neigh_obj.fit(training)
# classify the spectra
dist, ind = neigh_fit.kneighbors(target)

# %% codecell
labels = [ref_data_codes[i] for i in ind]

# %% codecell
# Re project the pixels onto the image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
tab10 = plt.get_cmap('tab10').colors
col_dict = {l:c for l, c in zip(barcodes, tab10)}
im_clust = np.zeros(max_sub.shape + (len(tab10[0]),))
for lab, i in zip(labels, pix_ind):
    x, y = i
    col = col_dict[lab[0]]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * max_sub[x,y]

len(labels)
np.sum(im_clust > 0)
# %% codecell
im_inches=10
ip.plot_image(im_clust, im_inches=im_inches)

# %% codecell
col_ordered = [col_dict[l] for l in barcodes]
fig, ax = ip.taxon_legend(
        taxon_names=barcodes,
        taxon_colors=col_ordered
        )


# %% md

# ==============================================================================
# ## Scaler
# ==============================================================================

# %% codecell
from sklearn.preprocessing import StandardScaler

pix_scaler = StandardScaler().fit(spec_pix)
spec_pix_scaled = pix_scaler.transform(spec_pix)

# %% codecell
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(spec_pix_scaled)
labels_kmeans.shape[0]
# %% codecell
n = 1000
labs = np.unique(labels_kmeans, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (labels_kmeans == l)
    spec_ = spec_pix_scaled[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.25,'alpha':0.25,'color':'r'})
    ax.set_title('cluster ' + str(l).zfill(2))

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
max_thresh = 0.05
im_inches=10

max_unsmooth = np.max(stack, axis=2)
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
ip.subplot_square_images([max_unsmooth, mask, max_masked], (1,3), im_inches=im_inches)

# %% codecell
spec_pix = stack_smooth[mask > 0]
spec_pix_norm = spec_pix / np.max(spec_pix, axis=1)[:,None]

# %% codecell
from sklearn.preprocessing import StandardScaler

pix_scaler = StandardScaler().fit(spec_pix)
spec_pix_scaled = pix_scaler.transform(spec_pix)


# %% codecell
# get reference spectra
ref_dir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/outputs/fig_5/2023_02_16_ecreference/HiPRFISH/seg_props'
fmt = '2023_02_16_ecreference_code_{}_fov_01_seg_props.csv'
barcodes = ['00001','00010','00100','01000','10000']
sci_names = ['Streptococcus','Neisseria','Veilonella','Lautropia','Corynebacterium']
avgint_cols = [str(i) for i in range(config['chan_start'],config['chan_end'])]

ref_spec = []
for bc in barcodes:
    fn = ref_dir + '/'+ fmt.format(bc)
    ref = pd.read_csv(fn)[avgint_cols].values
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
    print(ind)
    w_ = w.copy()
    w_[ind] = 0
    weights_top_n.append(w_)

for w, n in zip(weights_top_n, sci_names):
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
    ax.set_title(n)

# %% codecell
weights_pick = weights_sum_norm
input = spec_pix_scaled

weights_t = np.array(weights_pick).T
classif_mat = np.matmul(input, weights_t)
classifs_index = np.argmax(classif_mat, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell
n = 1000
labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    spec_ = spec_pix_norm[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.1,'color':'r'})
    ax.set_title(l)

# %% codecell
# Re project the pixels onto the image
# Reduce the white level
ul = 0.5
plot_intensities = max_unsmooth.copy()
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
tab10 = plt.get_cmap('tab10').colors
col_dict = {l:c for l, c in zip(labs_sort, tab10)}
im_clust = np.zeros(max_sub.shape + (len(tab10[0]),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell
im_inches=10
ip.plot_image(im_clust, im_inches=im_inches)
# output_dir = config['output_dir'] + '/test_classif/mat_mul'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# out_bn = output_dir + '/' + sn + '_zoom_classif_maxnorm_col'
# ip.save_png_pdf(out_bn)

# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )
# out_bn = out_dir + '/' + sn + '_taxon_legend'
# ip.save_png_pdf(out_bn)

# %% 
# set RGB
ulims = (0.45,0.45,0.45)

raws_shift_max = [np.max(r, axis=2) for r in raws_shift]
raws_shift_norm = [im / np.max(im) for im in raws_shift_max]
raws_shift_norm_adj = []
for im, ul in zip(raws_shift_norm, ulims):
    im[im>ul] = ul
    raws_shift_norm_adj.append(im/ul)
rgb = np.dstack(raws_shift_norm_adj)
ip.plot_image(rgb, im_inches=im_inches)

# %% codecell
############
# Zoom in on the veillonella cells

c=[1100,1200]
w=[50,50]

rgb_zoom = rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]

ip.plot_image(rgb_zoom, im_inches=im_inches)

# %% codecell
# Get the spectra 
dims=(10,5)
stack_zoom = stack_smooth[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
spec_zoom = stack_zoom[np.ones(w, dtype=np.bool)]
fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
# Plot veillonella reference spectrum 
for w, n in zip(weights_sum_norm, sci_names):
    if n=='Veilonella':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)

# %% codecell
############
# Zoom in on the super long veillonella cells

c=[430,250]
w=[50,50]

rgb_zoom = rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]

ip.plot_image(rgb_zoom, im_inches=im_inches)

# %% codecell
# Get the spectra 
dims=(10,5)
stack_zoom = stack_smooth[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
spec_zoom = stack_zoom[np.ones(w, dtype=np.bool)]
fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
# Plot veillonella reference spectrum 
for w, n in zip(weights_sum_norm, sci_names):
    if n=='Veilonella':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    elif n=='Neisseria':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)


# %% codecell
############
# Zoom in on the corynebacterium cells

c=[1435,675]
w=[40,50]

rgb_zoom = rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]

ip.plot_image(rgb_zoom, im_inches=im_inches)

# %% codecell
# Get the spectra 
dims=(10,5)
stack_zoom = stack_smooth[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
spec_zoom = stack_zoom[np.ones(w, dtype=np.bool)]
fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
# Plot veillonella reference spectrum 
for w, n in zip(weights_sum_norm, sci_names):
    if n=='Corynebacterium':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    elif n=='Lautropia':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)

# %% codecell
############
# Zoom in on more corynebacterium cells

c=[1880,425]
w=[120,200]

rgb_zoom = rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]

ip.plot_image(rgb_zoom, im_inches=im_inches)

# %% codecell
# Get the spectra 
dims=(10,5)
stack_zoom = stack_smooth[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
spec_zoom = stack_zoom[np.ones(w, dtype=np.bool)]
fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
# Plot veillonella reference spectrum 
for w, n in zip(weights_sum_norm, sci_names):
    if n=='Corynebacterium':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    elif n=='Lautropia':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)


# %% codecell
############
# Zoom in on other cells
c=[950,700]
w=[400,400]

rgb_zoom = rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]

ip.plot_image(rgb_zoom, im_inches=im_inches)

# %% codecell
# Get the spectra 
dims=(10,5)
stack_zoom = stack_smooth[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
spec_zoom = stack_zoom[np.ones(w, dtype=np.bool)]
fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
# Plot veillonella reference spectrum 
for w, n in zip(weights_sum_norm, sci_names):
    if n=='Neisseria':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Veilonella':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Streptococcus':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Lautropia':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Corynebacterium':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)

# %% codecell
# Get allfluor spectrum
barcode_length = 5
ref_chan_start = 32
allfluor_bc = [int(''.ljust(barcode_length, '1'))]
def get_reference_spectra(barcodes, bc_len, config):
    ref_dir = config['hipr_ref_dir']
    fmt = config['ref_files_fmt']
    if bc_len == 5:
        barcodes_str = [str(bc).zfill(5) for bc in barcodes]
        # barcodes_str = [str(bc).zfill(7) for bc in barcodes]
        barcodes_10bit = [bc[0] + '0' + bc[1] + '0000' + bc[2:] for bc in barcodes_str]
        # barcodes_10bit = [bc[0] + '0' + bc[1:4] + '00' + bc[4:] for bc in barcodes_str]
        barcodes_b10 = [int(str(bc),2) for bc in barcodes_10bit]
        st =  ref_chan_start + config['chan_start']
        en = ref_chan_start + config['chan_end']
        ref_avgint_cols = [i for i in range(st,en)]

        ref_spec = []
        for bc in barcodes_b10:
            fn = ref_dir + '/'+ fmt.format(bc)
            ref = pd.read_csv(fn, header=None)
            ref = ref[ref_avgint_cols].values
            ref_spec.append(ref)
    return ref_spec

allfluor_spec = get_reference_spectra(allfluor_bc, barcode_length, config)

def sum_normalize_ref(ref_spec):
    ref_sum_norm = []
    for r in ref_spec:
        r_ = r - np.min(r, axis=1)[:,None]
        ref_sum_norm.append(r_ / np.sum(r_, axis=1)[:,None])
    return [np.mean(r, axis=0) for r in ref_sum_norm]

allfluor_spec_norm = sum_normalize_ref(allfluor_spec)[0]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, allfluor_spec_norm[None,:], {'lw':1,'alpha':1,'color':'r'})

# %% codecell
# Get area under spec curves 
spec_zoom_areas = []
for sp in spec_zoom:
    spec_zoom_areas.append(np.sum(sp))
    # spec_zoom_areas.append(np.sum([(sp[i] + sp[i+1]) / 2 for i in range(sp.shape[0] - 1)]))

plt.scatter(spec_zoom_areas, spec_zoom_areas, s=0.05)

# %% codecell
# Adjust allfluor spectrum to subtract the background
sig = 2
ar_mean = np.mean(spec_zoom_areas)
ar_std = np.std(spec_zoom_areas)
thresh = ar_mean - sig*ar_std
bool_ar = np.array(spec_zoom_areas) < thresh


fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom[bool_ar,:], {'lw':0.1,'alpha':0.1,'color':'r'})

allfluor_spec_adj = allfluor_spec_norm * thresh
fsi.plot_cell_spectra(ax, allfluor_spec_adj[None,:], {'lw':1,'alpha':1,'color':'r'})



# %% codecell
# Subtract the allfluor spectrum
spec_zoom_adj = spec_zoom - allfluor_spec_adj[None,:]
spec_zoom_adj[spec_zoom_adj < 0] = 0
fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom_adj, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
# Get best weight for each laser


las_n_chan = [23,20,14]

top_n = 1
# weights_top_n = []
weights_oneperlaser_m = []
for wts in weights_max_norm:
    n = 0
    wt = []
    for n_chan in las_n_chan:
        n_1 = n + n_chan
        w_ = wts[n:n_1]
        l = len(w_)
        ind = np.argpartition(w_, l-top_n)[:l-top_n]
        w_p = w_.copy()
        w_p[ind] = 0
        wt += w_p.tolist()
        n = n_1
    weights_oneperlaser_m.append(wt)

weights_oneperlaser = []
for wt in weights_oneperlaser_m:
    wt /= np.sum(wt)
    print(np.sum(wt))
    weights_oneperlaser.append(wt)


for wt, n in zip(weights_oneperlaser, sci_names):
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, np.array(wt)[None,:], {'lw':1,'alpha':1,'color':'r'})
    ax.set_title(n)


# %% codecell
# Look at the classifier results for the zoom
input = spec_zoom_adj
weights_pick = weights_top_n

weights_t = np.array(weights_pick).T
classif_mat = np.matmul(input, weights_t)
classifs_index = np.argmax(classif_mat, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])
classif_mat[np.max(classif_mat, axis=1) > 0.06, :5]

# %% codecell
# Filter dim spectra 
thresh = 0.2
spec_zoom_max = np.max(spec_zoom_adj, axis=1)
bool_max = spec_zoom_max < thresh
classifs[bool_max] = 'None'

# %% codecell
# plot the results

ul = 0.5
c=[950,700]
w=[400,400]
stack_zoom_unsmooth = stack[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
sum_zoom = np.sum(stack_zoom_unsmooth, axis=2)
plot_intensities = sum_zoom.copy()
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(np.ones(sum_zoom.shape, dtype=np.bool))
# for each index draw the color
tab10 = plt.get_cmap('tab10').colors
# col_dict = {l:c for l, c in zip(labs_sort, tab10)}
col_dict = dict(zip(['Veilonella','Streptococcus','Neisseria','Lautropia','Corynebacterium'],tab10))
col_dict['None'] = (0.5,0.5,0.5)
im_clust = np.zeros(sum_zoom.shape + (len(tab10[0]),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell

im_inches=10
ip.plot_image(im_clust, im_inches=im_inches)

# %% codecell

labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )


# %% codecell
############################################
# run on whole image


# rgb_zoom = rgb[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
rgb_zoom = rgb.copy()

ip.plot_image(rgb_zoom, im_inches=im_inches)

# %% codecell
# Get the spectra 
dims=(10,5)
# stack_zoom = stack_smooth[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
stack_zoom = stack_smooth.copy()
# spec_zoom = stack_zoom[np.ones(w, dtype=np.bool)]
spec_zoom = stack_zoom[mask,:]
spec_zoom.shape
# fig, ax = ip.general_plot(dims=dims)
# fsi.plot_cell_spectra(ax, spec_zoom, {'lw':0.1,'alpha':0.1,'color':'r'})

# %% codecell
# Plot veillonella reference spectrum 
for w, n in zip(weights_sum_norm, sci_names):
    if n=='Neisseria':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Veilonella':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Streptococcus':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Lautropia':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)
    if n=='Corynebacterium':
        fig, ax = ip.general_plot(dims=(10,5))
        fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
        ax.set_title(n)

# %% codecell
# Get allfluor spectrum
barcode_length = 5
ref_chan_start = 32
allfluor_bc = [int(''.ljust(barcode_length, '1'))]
def get_reference_spectra(barcodes, bc_len, config):
    ref_dir = config['hipr_ref_dir']
    fmt = config['ref_files_fmt']
    if bc_len == 5:
        barcodes_str = [str(bc).zfill(5) for bc in barcodes]
        # barcodes_str = [str(bc).zfill(7) for bc in barcodes]
        barcodes_10bit = [bc[0] + '0' + bc[1] + '0000' + bc[2:] for bc in barcodes_str]
        # barcodes_10bit = [bc[0] + '0' + bc[1:4] + '00' + bc[4:] for bc in barcodes_str]
        barcodes_b10 = [int(str(bc),2) for bc in barcodes_10bit]
        st =  ref_chan_start + config['chan_start']
        en = ref_chan_start + config['chan_end']
        ref_avgint_cols = [i for i in range(st,en)]

        ref_spec = []
        for bc in barcodes_b10:
            fn = ref_dir + '/'+ fmt.format(bc)
            ref = pd.read_csv(fn, header=None)
            ref = ref[ref_avgint_cols].values
            ref_spec.append(ref)
    return ref_spec

allfluor_spec = get_reference_spectra(allfluor_bc, barcode_length, config)

def sum_normalize_ref(ref_spec):
    ref_sum_norm = []
    for r in ref_spec:
        r_ = r - np.min(r, axis=1)[:,None]
        ref_sum_norm.append(r_ / np.sum(r_, axis=1)[:,None])
    return [np.mean(r, axis=0) for r in ref_sum_norm]

allfluor_spec_norm = sum_normalize_ref(allfluor_spec)[0]

fig, ax = ip.general_plot(dims=(10,5))
fsi.plot_cell_spectra(ax, allfluor_spec_norm[None,:], {'lw':1,'alpha':1,'color':'r'})

# %% codecell
# Get area under spec curves 
spec_zoom_areas = []
for sp in spec_zoom:
    spec_zoom_areas.append(np.sum(sp))
    # spec_zoom_areas.append(np.sum([(sp[i] + sp[i+1]) / 2 for i in range(sp.shape[0] - 1)]))

# plt.scatter(spec_zoom_areas, spec_zoom_areas, s=0.05)

# %% codecell
# Adjust allfluor spectrum to subtract the background
sig = 1
ar_mean = np.mean(spec_zoom_areas)
ar_std = np.std(spec_zoom_areas)
thresh = ar_mean - sig*ar_std
bool_ar = np.array(spec_zoom_areas) < thresh


# fig, ax = ip.general_plot(dims=dims)
# fsi.plot_cell_spectra(ax, spec_zoom[bool_ar,:], {'lw':0.1,'alpha':0.1,'color':'r'})

allfluor_spec_adj = allfluor_spec_norm * thresh
fsi.plot_cell_spectra(ax, allfluor_spec_adj[None,:], {'lw':1,'alpha':1,'color':'r'})



# %% codecell
# Subtract the allfluor spectrum
spec_zoom_adj = spec_zoom - allfluor_spec_adj[None,:]
spec_zoom_adj[spec_zoom_adj < 0] = 0
fig, ax = ip.general_plot(dims=dims)
fsi.plot_cell_spectra(ax, spec_zoom_adj[::10], {'lw':0.1,'alpha':0.1,'color':'r'})



# %% codecell
# Look at the classifier results for the zoom
input = spec_zoom_adj
classif_mat = np.matmul(input, weights_t)
classifs_index = np.argmax(classif_mat, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])
classif_mat[np.max(classif_mat, axis=1) > 0.06, :5]

# %% codecell
# Filter dim spectra 
thresh = 0.1
spec_zoom_max = np.max(spec_zoom_adj, axis=1)
bool_max = spec_zoom_max < thresh
classifs[bool_max] = 'Unclassified'

# %% codecell
# get a grayscale image

ul = 0.25
gamma = 1.1

stack_zoom_unsmooth = stack.copy()
# stack_zoom_unsmooth = stack[c[0]:c[0]+w[0],c[1]:c[1]+w[1],:]
sum_zoom = np.sum(stack_zoom_unsmooth, axis=2)
plot_intensities = sum_zoom.copy()
# Gamma 
plot_intensities = plot_intensities**(1/gamma)
plot_intensities /= np.max(plot_intensities)
plot_intensities[plot_intensities > ul] = ul
plot_intensities /= ul

ip.plot_image(plot_intensities, cmap='gray', im_inches=10)

# %% codecell
# Project the classification

# Get the indices for each pixeel after masking
# pix_ind = np.argwhere(np.ones(sum_zoom.shape, dtype=np.bool))
pix_ind = np.argwhere(mask)
# for each index draw the color
tab10 = plt.get_cmap('tab10').colors
# col_dict = {l:c for l, c in zip(labs_sort, tab10)}
col_dict = dict(zip(['Veilonella','Streptococcus','Corynebacterium','Lautropia','Neisseria'],tab10))
col_dict['Unclassified'] = (0.5,0.5,0.5)
im_clust = np.zeros(sum_zoom.shape + (len(tab10[0]),))
classifs.shape
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * plot_intensities[x,y]

# %% codecell
# Plot the classification

im_inches=10
ip.plot_image(im_clust, im_inches=im_inches)

# %% codecell
# legend 

labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )

# %% codecell
# Look at spectra

n = 1000
labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    spec_ = spec_zoom_adj[bool,:]
    shp = spec_.shape[0]
    n_ = n if n < shp else shp
    inds = np.random.choice(spec_.shape[0], n_, replace=False)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.4,'alpha':0.1,'color':'r'})
    ax.set_title(l)


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

mega_dir = '../MeGAFISH'
config_mega_fn = mega_dir + '/config_mgefish.yaml'
with open(config_mega_fn, 'r') as f:
    config_mega = yaml.safe_load(f)

hiprmega_dir = '../HiPR_MeGA'
config_hiprmega_fn = hiprmega_dir + '/config.yaml'
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
    if out_fn: np.save(out_fn, hipr_resize)
    return hipr_resize

mega_res = config_mega['resolution']
hipr_res = config_hipr['resolution']
hipr_max_resize = resize_hipr(
        np.max(stack, axis=2), hipr_res, mega_res
        )

# %% codecell
# Get mega cell image
# config_hipr = config

mega_out_dir = mega_dir + '/' + config_mega['output_dir']

mega_raw_fn = mega_out_dir + '/' + config_mega['raw_fmt'].format(sample_name=sn)
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
        im_clust, hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
        )
hipr_maz_resize = resize_hipr(
        np.max(stack, axis=2), hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
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
im_inches=20
ul = 0.07
ll = 0.06
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
out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay'
ip.save_png_pdf(out_bn)


# %% codecell
# Show zoom
c=[2500,1450]
d=[1000,1000]
im_inches=10
ul = 0.075
ll = 0.065

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

hcr_zoom = hipr_classif_resize[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
fig, ax, cbar = ip.plot_image(
        hcr_zoom, scalebar_resolution=mega_res, im_inches=im_inches
        )
ax.imshow(sro_zoom)
out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay_zoom'
# ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Project only the veillonella
# ==============================================================================

# %% codecell
# adjust intensities
plot_intensities = hipr_max_resize.copy()
plot_intensities /= np.max(plot_intensities)


# %% codecell
# Get resized classifs list for pixel inds
barcodes_int = [int(bc) for bc in barcodes]
dict_sn_bc = dict(zip(sci_names, barcodes_int))
dict_sn_bc['Unclassified'] = 0
im_bc = np.zeros(max_sub.shape)
classifs_bc = [dict_sn_bc[sn] for sn in classifs]
pix_ind = np.argwhere(mask)
for bc, i in zip(classifs_bc, pix_ind):
    x, y = i
    # col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_bc[x,y] = bc
hipr_bc_resize = resize_hipr(
        im_bc, hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
        )

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
        plot_intensities_veil_mask * 0.1,
        plot_intensities_veil_mask * 0.4,
        plot_intensities_veil_mask * 0.7,
        plot_intensities_veil_mask
        ])
ip.plot_image(np.clip(rgb_veil, 0, 1))

# %% codecell
# Get spot RGB
spot_raw_coloc = spot_raw * mask_veil
rgb_spot_coloc = np.dstack([
        spot_raw_coloc * 1,
        spot_raw_coloc * 0,
        spot_raw_coloc * 1,
        spot_raw_coloc
        ])

ip.plot_image(np.clip(rgb_spot_coloc, 0, 1))

# %% codecell
# Get spot RGB
spot_raw_noncoloc = spot_raw * ~mask_veil
rgb_spot_noncoloc = np.dstack([
        spot_raw_noncoloc * 1,
        spot_raw_noncoloc * 1,
        spot_raw_noncoloc * 0,
        spot_raw_noncoloc
        ])

ip.plot_image(np.clip(rgb_spot_noncoloc, 0, 1))

# # %% codecell
# # Overlay
# rgb_spot_veil = np.clip(rgb_spot + rgb_veil, 0,1)
# ip.plot_image(rgb_spot_veil, im_inches=20)


# %% codecell
im_inches=20
ulg=1
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
out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_classif_spotraw_overlay_veilonella_zoom'
plt.figure(fig)
# ip.save_png_pdf(out_bn)




# %% md

# ==============================================================================
# ## Do spots based on simple threshold rather than clever filtering
# ==============================================================================

# %% codecell



# %% md

# ==============================================================================
# ## Spatial association between spots and pixels
# ==============================================================================

# Calculate actual spatial assoc

# %% codecell
# Get all the coords
r_um = 0.5

spot_coords = ref_pts_arr
hipr_mask_resize = resize_hipr(
        mask*1, hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
        )
pix_coords = np.argwhere(hipr_mask_resize)

r_pix = int(r_um / mega_res)

# %% codecell
# Get nearest neighbor classifier for pixels
# nbrs = NearestNeighbors(radius=r_pix).fit(pix_coords)
# # Search in radius around each spot
# inds = nbrs.radius_neighbors(spot_coords, return_distance=False)
# inds



# %% codecell
# Get resized classifs list for pixel inds
barcodes_int = [int(bc) for bc in barcodes]
dict_sn_bc = dict(zip(sci_names, barcodes_int))
dict_sn_bc['Unclassified'] = 0
im_bc = np.zeros(max_sub.shape)
classifs_bc = [dict_sn_bc[sn] for sn in classifs]
for bc, i in zip(classifs_bc, pix_ind):
    x, y = i
    # col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_bc[x,y] = bc
hipr_bc_resize = resize_hipr(
        im_bc, hipr_res, mega_res, dims=lrg.shape, ul_corner=ul_corner
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
    for k, v in sim_assoc_dict.items():
        sim_dict[k].append(v)

# %% md

# Plot

# %% codecell
# Plot simulated values for random
alpha=1
true_col='k'
true_lw=2
lw=1
dims=(3,2)
ft=7
nbins=100
n_pix = np.sum([v for k, v in true_assoc_dict.items()])
dict_bc_sciname = dict(zip(barcodes_int,sci_names))
for bc_tax in barcodes_int:
    sci_name = dict_bc_sciname[bc_tax]
    color = col_dict[sci_name]
    rand_counts = sim_dict[bc_tax]
    # Get fraction of total spots
    rand_frac = np.array(rand_counts) / n_pix
    # plot distribution of spot assignment
    nbins = np.unique(rand_frac).shape[0] // 4
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
    true_count = true_assoc_dict[bc_tax]
    true_frac = true_count / n_pix
    ax.plot([true_frac]*2, [0,0.75*ylims[1]], color=true_col, lw=true_lw)
    ax.set_title(sci_name)

# %% codecell
# Plot as boxplot
dims=[2.5,1.75]
xlab_rotation=25
marker='*'
marker_size=10
text_dist=0.05
ft=7
ylimadj = 0.03

barcodes_int_order = [100,1,10000,1000,10]

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
    if p_ > 0.001:
        t = str("p=" + str(p_))
    else:
        t = str("p<0.001")
    _ = ax.text(i+1, y, t, fontsize=ft, ha='center')
ax.set_xticklabels(xlab, rotation=xlab_rotation)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_spot_association_0_5um'
ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Spot association as fraction of spots rather than pixel count
# ==============================================================================

# %% codecell
# Function to generate spot x barcode boolean
# int_array = nb.types.int64[:]
@nb.njit()
def get_assocs_spot_arr(spot_coords, barcodes, hipr_bc_resize, r_pix=50):
    out_arr = np.zeros((spot_coords.shape[0], barcodes.shape[0]))
    for m, (x, y) in enumerate(spot_coords):
        for i in range(2*r_pix):
            for j in range(2*r_pix):
                x_i = int(x - r_pix + i)
                y_j = int(y - r_pix + j)
                # Get circle of pixels around spots
                if ((x-x_i)**2 + (y-y_j)**2)**(1/2) <= r_pix:
                    #Get barcodes
                    bc = np.int64(hipr_bc_resize[y_j, x_i])
                    if not bc == 0:
                        # Get array index for barcode
                        ind = np.argwhere(barcodes == bc)[0][0]
                        # If assoc not already found, add assoc
                        if out_arr[m,ind] == 0:
                            out_arr[m,ind] = 1
                        else:
                            pass
    return out_arr


# %% codecell
# Generate spot association array
import time
spot_assoc_arr = get_assocs_spot_arr(spot_coords, np.array(barcodes_int),
        hipr_bc_resize, r_pix)
spot_assoc_frac = np.sum(spot_assoc_arr, axis=0) / spot_assoc_arr.shape[0]
spot_assoc_frac

# %% codecell
# random distribution of spots
# iterate n times
n=1000
sim_list = []
for i in tqdm(range(n)):
    # Randomly select pixels
    i_sim = np.random.randint(0,pix_coords.shape[0], size=spot_coords.shape[0])
    # Generate counts for random selection
    sim_spot_coords = pix_coords[i_sim]
    sim_assoc_arr = get_assocs_spot_arr(sim_spot_coords, np.array(barcodes_int),
            hipr_bc_resize, r_pix=r_pix)
    sim_assoc_frac = np.sum(sim_assoc_arr, axis=0) / sim_assoc_arr.shape[0]
    sim_list.append(sim_assoc_frac)
sim_arr_spfrac = np.array(sim_list)
sim_arr_spfrac.shape

# %% codecell
# Plot as boxplot
dims=[2.5,1.75]
xlab_rotation=
marker='*'
marker_size=10
text_dist=0.075
ft=7
ylimadj = 0.075

# reorder taxa in plot
barcodes_int_order = [100,1,10000,1000,10]
bc_ind_order = [np.argwhere(np.array(barcodes_int) == bc)[0][0]
        for bc in barcodes_int_order]
sim_arr_ordered = np.hstack([sim_arr_spfrac[:,i][:,None] for i in bc_ind_order])

# create boxplot array
# n_pix = np.sum([v for k, v in true_assoc_dict.items()])
# sim_arr = [np.array(sim_dict[bc]) / n_pix for bc in barcodes_int_order]
# General plot
fig, ax = ip.general_plot(dims=dims, ft=ft)
# Plot simulation
boxplot = ax.boxplot(sim_arr_ordered, patch_artist=True, showfliers=False)
for m in boxplot['medians']:
    m.set_color('black')
for b in boxplot['boxes']:
    b.set_edgecolor('black')
    b.set_facecolor('white')
# Plot measured value
ys = []
xlab = []
for i, bc_tax in enumerate(barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    color = col_dict[sci_name]
    bc_ind = np.argwhere(np.array(barcodes_int) == bc_tax)[0][0]
    true_frac = spot_assoc_frac[bc_ind]
    _ = ax.plot(i+1, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals = sim_arr_spfrac[:,bc_ind]
    sim_mean = np.mean(sim_vals)
    if true_frac > sim_mean:
        r_ = sum(sim_vals > true_frac)
    else:
        r_ = sum(sim_vals < true_frac)
    p_ = r_ / n
    y_m = np.max(sim_vals)
    y = y_m + text_dist
    ys.append(y)
    if p_ > 0.001:
        t = str("p=" + str(p_))
    else:
        t = str("p<0.001")
    _ = ax.text(i+1, y, t, fontsize=ft, ha='center')
ax.set_xticklabels(xlab, rotation=xlab_rotation)
ylims = ax.get_ylim()
ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
out_dir = config['output_dir'] + '/pixel_classif_test'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_spot_association_frac_0_5um'
ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Pair correlation function
# ==============================================================================

# %% codecell
# Get area in pixels for whole classif
pix_bc_resize = hipr_bc_resize[hipr_bc_resize > 0]
count_pix = pix_bc_resize.shape[0]
# for each taxon Calculate the expected density
dict_bc_density = {}
for bc in barcodes_int:
    count_tax_pix = sum(pix_bc_resize == bc)
    dict_bc_density[bc] = count_tax_pix / count_pix
# count number of total spots
count_total_spots = spot_coords.shape[0]

# %% codecell
# Get radii in steps
max_r_um=20
dr_um=2

max_r_pix = max_r_um / config_mega['resolution']
dr_pix = dr_um / config_mega['resolution']
radii = np.arange(0, max_r_pix, dr_pix).astype(int)

# %% codecell
# function for shell mask
def createAnnularMask(dimx, dimy, center, big_radius, small_radius):

    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask_small = (distance_from_center >= small_radius)
    mask_big = (distance_from_center <= big_radius)

    return mask_small & mask_big

# %% codecell
r_size = radii[-1]
d = r_size * 2 + 1
annular_masks = []
for i in range(len(radii) - 1):
    r_1 = radii[i+1]
    r_0 = radii[i]
    annular_masks.append(createAnnularMask(d,d,[r_size,r_size],r_1,r_0))

for m in annular_masks:
    ip.plot_image(m)
    plt.show()
    plt.close()

# %% codecell
# maximum shell size
patch_size = annular_masks[-1].shape
im_size = hipr_bc_resize.shape
dict_bc_pcf = defaultdict(list)
# For each spot
for y,x in tqdm(spot_coords):
    # Get patch coordinates
    xmin, xmax = x - r_size, x + r_size + 1
    ymin, ymax = y - r_size, y + r_size + 1
    # Account for edge effects
    coords = []
    diffs = []
    for xy, im_s, p_s in zip([[ymin, ymax], [xmin, xmax]], im_size, patch_size):
        for v in xy:
            if v < 0:
                diffs.append(-v)
                coords.append(0)
            elif v > im_s:
                d = (v - im_s)
                diffs.append(d)
                coords.append(im_s)
            else:
                diffs.append(0)
                coords.append(v)
    # Get patch from image
    patch = hipr_bc_resize[coords[0]:coords[1],coords[2]:coords[3]]
    # for each taxon
    for bc in barcodes_int:
        # Get expected density
        rho = dict_bc_density[bc]
        pcf_tax = []
        # Search over a series of radii
        for m in annular_masks:
            # Correct for edge effects
            ends = patch_size[0]-diffs[1], patch_size[1]-diffs[3]
            m_corr = m[diffs[0]:ends[0],diffs[2]:ends[1]]
            # Extract all pixels from the shell
            pix_m = patch[m_corr]
            # Get area measure
            count_pix_m = np.sum(pix_m > 0)
            # Count taxon pixels within the shell
            count_pix_m_tax = np.sum(pix_m == bc)
            # Calculate pair correlation function
            calc = count_pix_m_tax / count_pix_m / rho
            pcf_tax.append(calc)
        dict_bc_pcf[bc].append(pcf_tax)



# %% codecell
dims=(5,5)
ft=12
line_col='k'
lw=1
alpha=0.5

for bc in barcodes_int:
    fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
    sci_name = dict_bc_sciname[bc]
    color = col_dict[sci_name]
    pcfs = dict_bc_pcf[bc]
    x = np.array([radii.tolist()[:-1] for _ in range(len(pcfs))]).T
    y = np.array(pcfs).T
    x.shape
    y.shape
    ax.plot(x, y, color=color, lw=lw, alpha=alpha)
    print(sci_name)
    plt.show()
    plt.close()


# %% codecell
dims=(5,5)
ft=12
line_col='k'
lw=2
alpha=1

fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
for bc in barcodes_int:
    sci_name = dict_bc_sciname[bc]
    color = col_dict[sci_name]
    pcfs = dict_bc_pcf[bc]
    x = np.array(radii.tolist()[:-1]) * config_mega['resolution']
    y = np.mean(pcfs, axis=0)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, label=sci_name)
ax.legend()
output_dir = config['output_dir'] + '/pair_correlation'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_pcf'
ip.save_png_pdf(out_bn)



# %% md

# Don't normalize by rho


# %% codecell
# maximum shell size
patch_size = annular_masks[-1].shape
im_size = hipr_bc_resize.shape
dict_bc_pcf = defaultdict(list)
# For each spot
for y,x in tqdm(spot_coords):
    # Get patch coordinates
    xmin, xmax = x - r_size, x + r_size + 1
    ymin, ymax = y - r_size, y + r_size + 1
    # Account for edge effects
    coords = []
    diffs = []
    for xy, im_s, p_s in zip([[ymin, ymax], [xmin, xmax]], im_size, patch_size):
        for v in xy:
            if v < 0:
                diffs.append(-v)
                coords.append(0)
            elif v > im_s:
                d = (v - im_s)
                diffs.append(d)
                coords.append(im_s)
            else:
                diffs.append(0)
                coords.append(v)
    # Get patch from image
    patch = hipr_bc_resize[coords[0]:coords[1],coords[2]:coords[3]]
    # for each taxon
    for bc in barcodes_int:
        # Get expected density
        rho = dict_bc_density[bc]
        pcf_tax = []
        # Search over a series of radii
        for m in annular_masks:
            # Correct for edge effects
            ends = patch_size[0]-diffs[1], patch_size[1]-diffs[3]
            m_corr = m[diffs[0]:ends[0],diffs[2]:ends[1]]
            # Extract all pixels from the shell
            pix_m = patch[m_corr]
            # Get area measure
            count_pix_m = np.sum(pix_m > 0)
            # Count taxon pixels within the shell
            count_pix_m_tax = np.sum(pix_m == bc)
            # Calculate pair correlation function
            calc = count_pix_m_tax / count_pix_m
            # calc = count_pix_m_tax / count_pix_m / rho
            pcf_tax.append(calc)
        dict_bc_pcf[bc].append(pcf_tax)



# %% codecell
dims=(5,5)
ft=12
line_col='k'
lw=1
alpha=0.5

for bc in barcodes_int:
    fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
    sci_name = dict_bc_sciname[bc]
    color = col_dict[sci_name]
    pcfs = dict_bc_pcf[bc]
    x = np.array([radii.tolist()[:-1] for _ in range(len(pcfs))]).T
    y = np.array(pcfs).T
    x.shape
    y.shape
    ax.plot(x, y, color=color, lw=lw, alpha=alpha)
    print(sci_name)
    plt.show()
    plt.close()


# %% codecell
dims=(5,5)
ft=12
line_col='k'
lw=2
alpha=1

fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
for bc in barcodes_int:
    sci_name = dict_bc_sciname[bc]
    color = col_dict[sci_name]
    pcfs = dict_bc_pcf[bc]
    x = np.array(radii.tolist()[:-1]) * config_mega['resolution']
    y = np.mean(pcfs, axis=0)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, label=sci_name)
ax.legend()
output_dir = config['output_dir'] + '/pair_correlation'
if not os.path.exists(output_dir): os.makedirs(output_dir)
out_bn = output_dir + '/' + sn + '_pcf_nonormalize'
ip.save_png_pdf(out_bn)


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

hipr_reseg_props.columns

# %% md

# ==============================================================================
# ## Plot segmentation
# ==============================================================================

# %% codecell
# Plot grayscale MGE
ip.plot_image(mega_cell, cmap='gray', im_inches=20)

# %% codecell
# plot grayscale hipr
ip.plot_image(plot_intensities, cmap='gray', im_inches=20)


# %% codecell
# get dict label to color
# dict_bc_sci = {int(bc):sci for bc, sci in zip(barcodes, sci_names)}
# tab10 = plt.get_cmap('tab10').colors
labs = hipr_reseg_props.label.values
genus = hipr_reseg_props.max_intensity.values
dict_bc_col = {int(bc):col_dict[sci] + (1,) 
    for bc,sci in zip(barcodes, sci_names)}
dict_bc_col[0] = (0,0,0,0)
dict_lab_col = {l:dict_bc_col[int(bc)] for l, bc in zip(labs, genus)}

# %% codecell
# recolor seg 
bboxes = hipr_reseg_props.bbox.values
dict_lab_bbox = dict(zip(labs, bboxes))
seg_classif_rgb = sf.seg_2_rgb(hipr_reseg, dict_lab_col, dict_lab_bbox)


# %% codecell
# Plot seg on grayscale and overlay MGE
im_inches=5

fig, ax, cbar = ip.plot_image(
    plot_intensities, 
    cmap='gray', 
    im_inches=im_inches,
    scalebar_resolution=mega_res
    )
ax.imshow(seg_classif_rgb)
ax.imshow(spot_raw_overlay)

out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_segclassif_spotraw_overlay'
plt.figure(fig)
dpi = min(plot_intensities.shape)/im_inches
ip.save_png_pdf(out_bn, dpi=dpi)
print('Wrote:',out_bn + '.png')


# %% codecell
# plot zoom image
c=[2500,1450]
d=[1000,1000]
im_inches=10

pi_zoom = plot_intensities[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
scr_zoom = seg_classif_rgb[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]
sro_zoom = spot_raw_overlay[c[0]: c[0]+d[0], c[1]: c[1]+d[1],:]

fig, ax, cbar = ip.plot_image(
    pi_zoom, 
    cmap='gray', 
    im_inches=im_inches,
    scalebar_resolution=mega_res
    )
ax.imshow(scr_zoom)
ax.imshow(sro_zoom)

out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_segclassif_spotraw_overlay_zoom'
plt.figure(fig)
dpi = min(pi_zoom.shape)/im_inches
ip.save_png_pdf(out_bn, dpi=dpi)
print('Wrote:',out_bn + '.png')

# %% codecell
# Get Veill only seg

labs_veil = labs[genus == 100]
dict_lab_col_veil = {}
for l, bc in zip(labs, genus):
    if bc == 100:
        col = dict_bc_col[bc]
    else:
        col = (0,0,0,0)
    dict_lab_col_veil[l] = col

seg_veil_rgb = sf.seg_2_rgb(hipr_reseg, dict_lab_col_veil, dict_lab_bbox)


# %% codecell
# plot veill only with grayscale and mge overlay

svr_zoom = seg_veil_rgb[c[0]: c[0]+d[0], c[1]: c[1]+d[1]]

fig, ax, cbar = ip.plot_image(
    pi_zoom, 
    cmap='gray', 
    im_inches=im_inches,
    scalebar_resolution=mega_res
    )
ax.imshow(svr_zoom)
ax.imshow(rgb_spot_coloc_zoom)
ax.imshow(rgb_spot_noncoloc_zoom)

out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_segclassif_spotraw_overlay_zoom_veil'
plt.figure(fig)
dpi = min(pi_zoom.shape)/im_inches
ip.save_png_pdf(out_bn, dpi=dpi)
print('Wrote:',out_bn + '.png')


# %% md

# ==============================================================================
# ## nearest neighbor histograms
# ==============================================================================

# %% codecell
# For each taxon
if 0 not in barcodes_int:
    barcodes_int.append(0)

dict_bc_dists = {}
for bc in barcodes_int:
    # Get cell coords for taxon
    bool_bc = hipr_reseg_props.max_intensity == bc
    tax_centroid = [list(c) for c in hipr_reseg_props.loc[bool_bc,'centroid'].values]
    # Get nearest neighbor cell distance for each spot
    nbrs = NearestNeighbors(n_neighbors=1).fit(tax_centroid)
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

dict_bc_sciname = dict(zip(barcodes_int, sci_names))
dict_bc_sciname[0] = 'Unclassified'


fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
for bc in barcodes_int:
    sci_name = dict_bc_sciname[bc]
    color = col_dict[sci_name]
    dists = dict_bc_dists[bc]
    # hist, bin_edges = np.histogram(dists, bins=bins)
    hist, bin_edges = np.histogram(dists, bins=bins, density=True)
    x = ip.get_line_histogram_x_vals(bin_edges) * config_mega['resolution']
    ax.plot(x, hist, color=color, lw=lw, alpha=alpha, label=sci_name)
ax.set_xlim(xlims[0],xlims[1])
ax.legend()

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
    for bc in barcodes_int:
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

for bc in barcodes_int:
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
    # output_dir = config['output_dir'] + '/reclassify'
    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # out_bn = output_dir + '/' + sn + '_nn_histogram_' + sci_name
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
barcodes_int
barcodes_order = [100,1000,10,10000,1,0]
for bc in barcodes_order:
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
for i, bc_tax in enumerate(barcodes_order):
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

np.median(sim_z[3,:])


# %% codecell
# Plot z score
dims=[2.1,1.83]
xlab_rotation=45
pval_rotation=60
marker='.'
marker_size=15
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
        sim_z.T, patch_artist=True, showfliers=False,
        boxprops=dict(facecolor=box_col, color=box_line_col),
        capprops=dict(color=box_line_col),
        whiskerprops=dict(color=box_line_col),
        medianprops=dict(color=box_line_col),
      )
# boxplot = ax.boxplot(
#         sim_z[3,:][:,None], patch_artist=True, showfliers=False,
#         boxprops=dict(facecolor=box_col, color=box_line_col),
#         capprops=dict(color=box_line_col),
#         whiskerprops=dict(color=box_line_col),
#         medianprops=dict(color=box_line_col),
#       )
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
for i, bc_tax in enumerate(barcodes_order):
# for i, bc_tax in zip(ind_order, barcodes_int_order):
    sci_name = dict_bc_sciname[bc_tax]
    xlab.append(sci_name)
    try:
        color = col_dict[sci_name]
    except:
        continue
    true_frac = meas_z[i]
    # true_frac = true_count / n_cells
    # _ = ax.plot(x, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals_i = sim_vals[i,:]
    # sim_vals = sim_arr[:,i,h] / n_cells
    sim_mean = np.mean(sim_vals)
    if sci_name == 'Corynebacterium':
        print(sim_mean)
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
ax.set_yticks(ticks=[-5,0,5,10], labels=[])
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['right'].set_color('none')

# ylims = ax.get_ylim()
# ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_seg_nn_zscore_association_0_5um'
# ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Bar plot fraction of spots and cells assoc
# ==============================================================================

# %% codecell
# Frac spots assoc with taxon
dims=[2.1,0.7]
ft=6
line_col = 'k'
width=0.4

sci_name_order = [dict_bc_sciname[bc] for bc in barcodes_order]
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
out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_bar_seg_nn_frac_spot_association_0_5um'
# ip.save_png_pdf(out_bn)


# %% codecell
tax_counts = np.array([
        hipr_reseg_props[hipr_reseg_props.max_intensity == bc].shape[0]
        for bc in barcodes_order
        ])

meas_frac_cell = meas_vals / tax_counts

# %% codecell
# Frac spots assoc with taxon
dims=[2.1,0.7]
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
ax.set_yticks(ticks=[0,0.02,0.04], labels=[])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
out_dir = config['output_dir'] + '/reclassify'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_bar_seg_nn_frac_cell_association_0_5um'
# ip.save_png_pdf(out_bn)


# # %% codecell
# # Get nearest neighbors
# h_max_um = 10
# h_step_um = 0.5
# n_neighbors=1
#
# # Convert um measure to pixels
# h_max_pix = h_max_um / config_mega['resolution']
# h_step_pix = h_step_um / config_mega['resolution']
# # Get a range of distances to measure how many associations
# hs = np.arange(0, h_max_pix, h_step_pix)
# hs_um = hs * config_mega['resolution']
# # Save the locations of each set of species
# barcodes_unq = hipr_reseg_props.max_intensity.unique().astype(int).tolist()
# barcodes_unq.remove(0)
# dict_cell_coords = {}
# for bc in barcodes_unq:
#     bc_bool = hipr_reseg_props.max_intensity == bc
#     reseg_centr = hipr_reseg_props.loc[bc_bool, 'centroid'].values
#     reseg_coords = np.array([list(c) for c in reseg_centr])
#     dict_cell_coords[bc] = reseg_coords
#
#
# # %% codecell
# # Measure the sample
# # get the spot coordinates and create a nneighbors object from them
# spot_coords_float = np.array([list(c) for c in spot_props_shift.centroid.values])
# nbrs = NearestNeighbors(n_neighbors=n_neighbors)
# nbrs.fit(spot_coords_float)
# # initialize a dict to save the count values for each distance
# dict_h = {}
# for bc in barcodes_unq:
#     reseg_coords = dict_cell_coords[bc]
#     dists, inds = nbrs.kneighbors(reseg_coords)
#     dict_h[bc] = []
#     for h in hs:
#         count = np.sum(dists < h)
#         dict_h[bc].append(count)
#
#
# # %% codecell
# # simulate random distribution of spots
# n = 1000
#
# dict_sim = defaultdict(list)
# for i in tqdm(range(n)):
#     # Get random pixels
#     i_sim = np.random.randint(
#             0, pix_coords.shape[0], size=spot_coords_float.shape[0]
#             )
#     sim_spot_coords = pix_coords[i_sim]
#     nbrs = NearestNeighbors(n_neighbors=n_neighbors)
#     nbrs.fit(sim_spot_coords)
#     list_bc = []
#     for bc in barcodes_unq:
#         reseg_coords = dict_cell_coords[bc]
#         dists, inds = nbrs.kneighbors(reseg_coords)
#         list_h = []
#         for h in hs:
#             count = np.sum(dists < h)
#             list_h.append(count)
#         dict_sim[bc].append(list_h)
#
#
# np.array(dict_sim[10]).shape
#
# # %% codecell
# # Plot simulation min max vs sample
# dims=(5,3)
# sim_lw=1
# sim_col = 'k'
# sim_alpha=0.5
# sample_color='r'
# sample_lw=2
#
# for i, bc in enumerate(barcodes_unq[1:]):
#     fig, ax = ip.general_plot(dims=dims)
#     sim_bc = dict_sim[bc]
#     # sim_bc = sim_arr[:,i,:]
#     sim_bc_ll = np.min(sim_bc, axis=0)
#     sim_bc_ul = np.max(sim_bc, axis=0)
#     ax.plot(hs_um, sim_bc_ll, color=sim_col, lw=sim_lw)
#     ax.plot(hs_um, sim_bc_ul, color=sim_col, lw=sim_lw)
#     ax.plot(hs_um, dict_h[bc], lw=sample_lw, color=sample_color)
#     print(dict_bc_sciname[bc])
#     plt.show()
#     plt.close()
#
#
# # %% codecell
# # Plot simulation array vs sample
# dims=(5,3)
# sim_lw=0.5
# sim_col = 'k'
# sim_alpha=0.1
# sample_color='r'
# sample_lw=2
#
# len(spot_coords)
# for i, bc in enumerate(barcodes_unq[1:]):
#     fig, ax = ip.general_plot(dims=dims)
#     sim_bc = dict_sim[bc]
#     # sim_bc = sim_arr[:,i,:]
#     for h in sim_bc:
#         ax.plot(hs_um, h, lw=sim_lw, alpha=sim_alpha, color=sim_col)
#     ax.plot(hs_um, dict_h[bc], lw=sample_lw, color=sample_color)
#     print(dict_bc_sciname[bc])
#     plt.show()
#     plt.close()
#
#
# # %% codecell
# # Plot as boxplot for a given h
# h = 4
#
# # h = int(h_um / config_mega['resolution'])
# h_um = hs[h]*config_mega['resolution']
# dims=[2.75,1.5]
# xlab_rotation=45
# pval_rotation=60
# marker='*'
# marker_size=10
# text_dist=0.1
# ft=7
# ylimadj = 0.1
# true_frac_llim = 0
# line_col = 'k'
# box_col = 'w'
#
# # barcodes_int_order = [100,1,10000,1000,10]
# #
#
# # Order barcodes by p value
# pvals_pos, pvals_neg = [], []
# bc_pos, bc_neg = [], []
# for i, bc_tax in enumerate(barcodes_unq):
#     print(bc_tax)
#     true_count = dict_h[bc_tax][h]
#     print(true_count)
#     n_cells = dict_cell_coords[bc_tax].shape[0]
#     print(n_cells)
#     true_frac = true_count / n_cells
#     print(true_frac)
#     sim_vals = sim_arr[:,i,h] / n_cells
#     sim_mean = np.mean(sim_vals)
#     print(sim_mean)
#     if true_frac > sim_mean:
#         r_ = sum(sim_vals > true_frac)
#         pvals_pos.append(r_ / n)
#         bc_pos.append(bc_tax)
#     else:
#         r_ = sum(sim_vals < true_frac)
#         pvals_neg.append(r_ / n)
#         bc_neg.append(bc_tax)
#
# barcodes_pos_order = [x for _, x in sorted(zip(pvals_pos, bc_pos), reverse=False)]
# barcodes_neg_order = [x for _, x in sorted(zip(pvals_neg, bc_neg), reverse=True)]
# barcodes_int_order = barcodes_pos_order + barcodes_neg_order
# barcodes_int_order = [int(bc) for bc in barcodes_int_order]
#
# # %% codecell
# # Order the indices of the barcodes
# ind_order = []
# for bc in barcodes_int_order:
#     ind = np.argwhere(np.array(barcodes_unq) == bc)[0][0]
#     ind_order.append(ind)
#
# # barcodes_unq[12]
# dict_bc_sciname = dict(zip(barcodes_int,sci_names))
#
# # create boxplot array
# # n_pix = np.sum([v for k, v in true_assoc_dict.items()])
# # sim_arr = [np.array(sim_dict[bc]) / n_pix for bc in barcodes_int_order]
# sim_arr_h = []
# for i, bc in zip(ind_order, barcodes_int_order):
#     n_cells = dict_cell_coords[bc].shape[0]
#     sim_arr_h.append(sim_arr[:,i,h]/n_cells)
# # General plot
# fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
# # Plot simulation
# boxplot = ax.boxplot(
#         sim_arr_h, patch_artist=True, showfliers=False,
#         boxprops=dict(facecolor=box_col, color=line_col),
#         capprops=dict(color=line_col),
#         whiskerprops=dict(color=line_col),
#         medianprops=dict(color=line_col),
#       )
# # for m in boxplot['medians']:
# #     m.set_color(line_col)
# # for b in boxplot['boxes']:
# #     b.set_edgecolor(line_col)
# #     b.set_facecolor(box_col)
# col_dict
# # Plot measured value
# ys = []
# xlab = []
# x = 1
# for i, bc_tax in zip(ind_order, barcodes_int_order):
#     sci_name = dict_bc_sciname[bc_tax]
#     print(sci_name)
#     xlab.append(sci_name)
#     try:
#         color = col_dict[sci_name]
#     except:
#         continue
#     true_count = dict_h[bc_tax][h]
#     n_cells = dict_cell_coords[bc_tax].shape[0]
#     true_frac = true_count / n_cells
#     _ = ax.plot(x, true_frac, marker=marker, ms=marker_size, color=color)
#     # Plot p value
#     sim_vals = sim_arr[:,i,h] / n_cells
#     sim_mean = np.mean(sim_vals)
#     if true_frac > sim_mean:
#         # number of simulations with value greater than observed
#         r_ = sum(sim_vals > true_frac)
#     else:
#         # number of simulations with value less than observed
#         r_ = sum(sim_vals < true_frac)
#     # P value
#     p_ = r_ / n
#     # Get text location
#     q1,q3 = np.quantile(sim_vals, [0.25,0.75])
#     q4 = q3 + 1.5 * (q3 - q1)
#     # y_m = np.max(sim_vals)
#     # y = y_m if y_m > true_frac else true_frac
#     y = q4 if q4 > true_frac else true_frac
#     y += text_dist
#     ys.append(y)
#     if true_frac < true_frac_llim:
#         t = ''
#     elif (p_ > 0.05):
#         t = ''
#     elif (p_ > 0.001) and (p_ <= 0.05):
#         t = str("p=" + str(p_))
#     else:
#         t = str("p<0.001")
#     _ = ax.text(x, y, t, fontsize=ft, ha='left',va='bottom', rotation=pval_rotation, rotation_mode='anchor',
#             color=line_col)
#     x+=1
# ax.set_xticklabels([], rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
# # ax.set_xticklabels(xlab, rotation=xlab_rotation, ha='right', va='top', rotation_mode='anchor')
# ax.tick_params(axis='x',direction='out')
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
#
# ylims = ax.get_ylim()
# ax.set_ylim(ylims[0], np.max(ys) + ylimadj)
# print(str(h_um) + ' micrometers')
# # out_dir = config['output_dir'] + '/spatial_assoc'
# # if not os.path.exists(out_dir): os.makedirs(out_dir)
# # out_bn = out_dir + '/' + sn + '_spot_association_frac_cells_dist_1um'
# # ip.save_png_pdf(out_bn)

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
for i, bc in enumerate(barcodes_unq):
    fig, ax = ip.general_plot(dims=dims)
    sim_bc = sim_arr[:,i,:]
    print(np.mean(sim_arr[:,i,2]))
    x = hs * config_mega['resolution']
    for h in sim_bc:
        ax.plot(x, h, lw=sim_lw, alpha=sim_alpha, color=sim_col)
    ax.plot(x, dict_h[bc], lw=sample_lw, color=sample_color)
    print(dict_h[bc][2])
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
    # true_frac = true_count / n_cells
    true_frac = true_count
    # sim_vals = sim_arr[:,i,h] / n_cells
    sim_vals = sim_arr[:,i,h]
    sim_mean = np.mean(sim_vals)
    print(sim_mean)
    print(true_count)
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

# %% codecell

# Order the indices of the barcodes
ind_order = []
for bc in barcodes_int_order:
    ind = np.argwhere(np.array(barcodes_unq) == bc)[0][0]
    ind_order.append(ind)

# barcodes_unq[12]
dict_bc_sciname = dict(zip(barcodes_int, sci_names))

# create boxplot array
# n_pix = np.sum([v for k, v in true_assoc_dict.items()])
# sim_arr = [np.array(sim_dict[bc]) / n_pix for bc in barcodes_int_order]
sim_arr_h = []
for i, bc in zip(ind_order, barcodes_int_order):
    n_cells = dict_cell_coords[bc].shape[0]
    sim_arr_h.append(sim_arr[:,i,h])
    # sim_arr_h.append(sim_arr[:,i,h]/n_cells)
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
    true_frac = true_count
    # true_frac = true_count / n_cells
    _ = ax.plot(x, true_frac, marker=marker, ms=marker_size, color=color)
    # Plot p value
    sim_vals = sim_arr[:,i,h]
    # sim_vals = sim_arr[:,i,h] / n_cells
    sim_mean = np.mean(sim_vals)
    print(sim_mean)
    print(true_count)
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
# out_dir = config['output_dir'] + '/spatial_assoc'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# out_bn = out_dir + '/' + sn + '_spot_association_frac_cells_dist_1um'
# ip.save_png_pdf(out_bn)




# %% md

# ==============================================================================
# ## Nearest neighbor spots to cells spatial assoc
# ==============================================================================

# %% codecell
# Do nearest neighbor spots to cells
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
dims=[3,3]
# dims=[1.,0.75]
sim_lw=0.5
sim_col = 'k'
sim_alpha=0.1
# sample_color='r'
sample_lw=2
save_list = ['Veillonella','Leptotrichia','Fusobacterium']
ft = 12

hs_um = hs*config_mega['resolution']
len(spot_coords)
for i, bc in enumerate(barcodes_int_order):
    sci_name = dict_bc_sciname[bc]
    # if sci_name in save_list:
    color = col_dict[sci_name]
    fig, ax = ip.general_plot(dims=dims, ft=ft)
    sim_bc = sim_arr[:,i,:]
    # sim_bc =
    for h in sim_bc:
        ax.plot(hs_um, h, lw=sim_lw, alpha=sim_alpha, color=sim_col)
    ax.plot(hs_um, dict_h[bc], lw=sample_lw, color=color)
    # ax.plot(hs_um, dict_h[bc], lw=sample_lw, color=sample_color)
    # if not i == 3:
    # ax.set_xticks(ticks=[0,2,4,6,8], labels=[])
    # ax.set_yticks(ticks=[0,50,100,150], labels=[])
    print(dict_bc_sciname[bc])
        # out_dir = config['output_dir'] + '/spatial_assoc'
        # if not os.path.exists(out_dir): os.makedirs(out_dir)
        # out_bn = out_dir + '/' + sn + '_cell_association_lag_plot_' + sci_name
        # ip.save_png_pdf(out_bn)
        # print('saved')

    plt.show()
    plt.close()


# %% codecell
# Plot as boxplot for a given h
h = 1

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
dict_bc_sciname = dict(zip(barcodes_int, sci_names))

# create boxplot array
# n_pix = np.sum([v for k, v in true_assoc_dict.items()])
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
ip.save_png_pdf(out_bn)


# %% md

# ==============================================================================
# ## Stacked bar plots
# ==============================================================================

# %% codecell
# Get 10 nearest neighbors for spots
n_neighbors = 20

cell_coords_tup = hipr_reseg_props.loc[:,'centroid'].values
cell_coords = np.array([list(c) for c in cell_coords_tup])
cell_bc = np.array(hipr_reseg_props.max_intensity.astype(int).tolist())
nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(cell_coords)
dists, inds = nbrs.kneighbors(spot_coords)

inds[:5]

# %% codecell
# Use indices to get taxon identities
# Get taxon counts for each closest neighbor
bc_order = [100,1,10,1000,10000,0]
tax_counts_nn =[]
for i in range(n_neighbors):
    bcs_i = cell_bc[inds[:,i]]
    unq, bcs_i_count = np.unique(bcs_i, return_counts=True)
    dict_count = dict(zip(unq, bcs_i_count))
    count_list = []
    for bc in bc_order:
        try:
            count_list.append(dict_count[bc])
        except:
            count_list.append(0)
    tax_counts_nn.append(count_list)

# %% codecell
# assemble for stacke dbar plot
dims=[3,3]
ft=12
line_col = 'k'
tax_counts_nn = np.array(tax_counts_nn)
fig, ax = ip.general_plot(dims=dims, ft=ft, col=line_col)
ntax = tax_counts_nn.shape[1]
x = np.arange(n_neighbors)
for i, bc in enumerate(bc_order):
    bottom = np.zeros(n_neighbors)
    # if i > 0:
    j = i - 1
    while j >= 0:
        bottom += tax_counts_nn[:,j]
        j -= 1
    if bc == 0:
        color=(0.5,0.5,0.5)
        sci_name='No Barcode'
    else:
        sci_name = dict_bc_sciname[bc]
        color = col_dict[sci_name]
    ax.bar(x, tax_counts_nn[:,i], bottom=bottom, color=color, label=sci_name)

out_dir = config['output_dir'] + '/nn_stacked_bar'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + 'nn_stacked_bar'
ip.save_png_pdf(out_bn)

# %% md

# ==============================================================================
# ## Plot spot density on map
# ==============================================================================

# %% codecell
# Get cell
cell = hipr_max_resize
cell_pix_values = np.sort(np.ravel(cell))

# %% codecell
# PLot intensities
thresh_cell=0.095
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
r_um = 10
r_pix = int(r_um / config_mega['resolution'])




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


# # # %% codecell
# # manually draw mask in fiji
# edge_mask_fn = config['output_dir'] + '/edge_mask/' + sn + 'edgemask.tif'
#
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
#
#
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
#
# # %% codecell
# # plot edge correction
# cell_edge_map = np.zeros(cell.shape)
# cell_edge_map[pix_inds] = area_list
# fig, ax, cbar = ip.plot_image(cell_edge_map)
# background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
# background = np.dstack(background)
# ax.imshow(background)

# %% codecell
# Calculate spot densities at subset of pixels
subset=2
chan_densities = []
# spot_coords_sub = np.array(spot_coords)[::subset]
# area_list_sub = np.array(area_list)[::subset]
props = mega_shift
for s_ch in config_mega['spot_seg']['channels']:
    print(s_ch)
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
    pix_density = np.array(pix_counts) / area_circle
    # pix_density = np.array(pix_counts) / np.array(area_list)
    chan_densities.append(pix_density)


# %% codecell
# Plot
clims = (0, np.max(chan_densities))
for s_ch, pix_density in zip(config_mega['spot_seg']['channels'][:1], chan_densities):
    print(s_ch)
    cell_spot_density = np.zeros(cell.shape)
    cell_spot_density[pix_inds] = pix_density
    fig, ax, cbar = ip.plot_image(
            cell_spot_density,
            cmap='cividis',
            scalebar_resolution=config_mega['resolution'],
            clims=clims,
            cbar_ori='vertical'
            )
    background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
    background = np.dstack(background)
    ax.imshow(background)
    plt.figure(fig)
    out_dir = config['output_dir'] + '/density_maps'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    output_bn = out_dir + '/' + sn + '_spot_density_map_chan_' + str(s_ch)
    ip.save_png_pdf(output_bn)
    plt.figure(cbar)
    out_dir = config['output_dir'] + '/density_maps'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    output_bn = out_dir + '/' + sn + '_spot_density_map_chan_' + str(s_ch) + '_cbar'
    ip.save_png_pdf(output_bn)
    plt.show()
    plt.close()


# %% codecell
# Threshold high density areas
thresh_density = 0.035
bool_density = cell_spot_density > thresh_density
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
ip.save_png_pdf(output_bn)


# %% codecell
# Get composition in high vs low densiy areas
hipr_reseg_hd = hipr_reseg * (cell_spot_density > thresh_density)
fig, ax, cbar = ip.plot_image(
        hipr_reseg_hd > 0,
        cmap='gray',
        scalebar_resolution=config_mega['resolution']
        )

labels_hd = np.unique(hipr_reseg_hd)
bool_labels_hd = hipr_reseg_props.label.isin(labels_hd)
bool_labels_ld = ~hipr_reseg_props.label.isin(labels_hd)

# %% codecell
bool_density_h = np.array(cell_spot_density) > thresh_density
bcs_high_density = hipr_reseg_props.loc[
        bool_labels_hd,
        'max_intensity'
        ].value_counts().sort_index()
bool_density_l = np.array(cell_spot_density) <= thresh_density
bcs_low_density = hipr_reseg_props.loc[
        bool_labels_ld,
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
ip.save_png_pdf(output_bn)



# %% codecell
