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
# javabridge.start_vm(class_path=bioformats.JARS)
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
raws_max_norm = [im / np.max(im) for im in raws_max]
raws_max_norm_adj = []
for im, ul in zip(raws_max_norm, ulims):
    im[im>ul] = ul
    raws_max_norm_adj.append(im/ul)
ip.plot_image(np.dstack(raws_max_norm_adj), im_inches=im_inches)
# plt.savefig(out_dir + '/' + sn + '_rgb.png', dpi=raws[0].shape//(im_inches*2))
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
out_dir = config['output_dir'] + '/pixel_classif_test'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_spot_association_0_5um'
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


# %% codecell
