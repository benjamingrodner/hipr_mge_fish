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
import math
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import re
from collections import defaultdict
import javabridge
import bioformats
from tqdm import tqdm
import aicspylibczi as aplc
from sklearn.neighbors import NearestNeighbors




# %% md

# Move to the workdir

# %% codecell
# Absolute path
project_workdir = '/workdir/bmg224/manuscripts/mgefish/code/fig_1/notebooks'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory


# %% md

# Define paths

# %% codecell
data_dir = '../../../data/fig_1'
output_dir = '../../../outputs/fig_1'
pipeline_path = '../..'


# %% codecell
# params
resolution = 0.04
channel_names = ['rRNA','GFP_DNA']

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%load_ext autoreload
%autoreload 2

sys.path.append(pipeline_path + '/functions')
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
# get samples
ext = '.czi'
filenames = glob.glob(data_dir + '/*' + ext)
sample_names = [re.sub(ext,'',os.path.split(fn)[1]) for fn in filenames]
sample_names

# %% codecell
javabridge.start_vm(class_path=bioformats.JARS)

# %% codecell
# pick sample
i = 3
sn = sample_names[i]
fn = filenames[i]
raw = bioformats.load_image(fn)
raw_chans = [raw[:,:,i] for i in range(raw.shape[2])]

# %% codecell
# metadata
raw_aplc = aplc.CziFile(fn)

for elem in raw_aplc.meta.iter():
    if len(elem.attrib) > 0:
        print(elem.tag, elem.attrib)

# %% codecell
# Load cell morphology info
cell_props_fn = output_dir + '/segmentation/cell_seg_props/' + sn + '/' + sn + '_chan_0_cell_seg_props.csv'
cell_props = pd.read_csv(cell_props_fn)
cell_props.shape

# %% codecell
# Look at cell segmentation
cell_seg_fn = output_dir + '/segmentation/cell_seg/' + sn + '/' + sn + '_chan_0_cell_seg.npy'
cell_seg = np.load(cell_seg_fn)
# %% codecell
cell_seg_rgb = ip.seg2rgb(cell_seg)
ip.plot_image(cell_seg_rgb)

# %% codecell
c = [500, 500]
d = [500,500]
cell_seg_rgb_zoom = cell_seg_rgb[c[0]: c[0]+d[0], c[1]: c[1]+d[1], :]
ip.plot_image(cell_seg_rgb_zoom)

# %% codecell
import math

def magnitude(vector):
    return math.sqrt(sum(a * a for a in vector))


# %% codecell
a = [2,2]
b = [0,-1]

abs(np.dot(a,b) / magnitude(b))


# %% codecell
cell_props.columns

cell_props.loc[:5,'orientation']

# %% codecell
# Get spot props
spot_props_fn = output_dir + '/segmentation/spot_analysis/' + sn + '/' + sn + '_cellchan_0_spotchan_1_spot_seg_cid.csv'
spot_props = pd.read_csv(spot_props_fn)
spot_props.shape

# %% codecell
# Build cell dictionary to speed up lookups
cell_dict = {}
for i, c in cell_props.iterrows():
    l = c.label
    cell_dict[l] = {}
    cell_dict[l]['cent'] = np.array(eval(c.centroid))
    cell_dict[l]['ori'] = c.orientation
    cell_dict[l]['len'] = c.major_axis_length
    cell_dict[l]['wid'] = c.minor_axis_length
    cell_dict[l]['ecc'] = c.eccentricity

# %% codecell
# for each spot
pole_fracs = []
for i, sp in tqdm(spot_props.iterrows()):
    # get coords
    # get cell
    cid = sp.cell_id
    if cid > 0:
        cell = cell_dict[cid]
        # get orientation
        ori = cell['ori']
        # Get cell axis vector
        if abs(ori) == (math.pi / 2):
            vcell = [0,1]
        else:
            vcell = [1, math.tan(ori)]
        vcell = np.array(vcell)
        # Get spot axis vector
        sp_coords = np.array([sp['centroid-0'], sp['centroid-1']])
        vspot = sp_coords - cell['cent']
        # get dot product for spot
        dist = abs(np.dot(vspot, vcell) / magnitude(vcell))
        # Normalize by cell length
        cell_len = cell['len'] / 2
        pole_fracs.append(dist / cell_len)

# %% codecell
n, bins, patches = plt.hist(pole_fracs, bins=200)


# %% codecell
hist = plt.hist(cell_props.eccentricity.values, bins=100)

# %% codecell
# only look at cells with eccentricity
ecc_thresh=0.9

pole_fracs = []
for i, sp in tqdm(spot_props.iterrows()):
    # get coords
    # get cell
    cid = sp.cell_id
    if cid > 0:
        cell = cell_dict[cid]
        if cell['ecc'] > ecc_thresh:
            # get orientation
            ori = cell['ori']
            # Get cell axis vector
            if abs(ori) == (math.pi / 2):
                vcell = [0,1]
            else:
                vcell = [1, math.tan(ori)]
            vcell = np.array(vcell)
            # Get spot axis vector
            sp_coords = np.array([sp['centroid-0'], sp['centroid-1']])
            vspot = sp_coords - cell['cent']
            # get dot product for spot
            dist = abs(np.dot(vspot, vcell) / magnitude(vcell))
            # Normalize by cell length
            cell_len = cell['mal'] / 2
            pole_fracs.append(dist / cell_len)

# %% codecell
dims=(1.5,1)
ft=6
xlims=(0,1.25)
fig, ax = ip.general_plot(dims=dims, ft=ft)
n, bins, patches = ax.hist(pole_fracs, bins=25, color=(1,0,1))
ax.set_xticks(ticks=[0,0.5,1], labels=[])
ax.set_yticks(ticks=[0,1000,2000,3000], labels=[])
ax.set_xlim(xlims[0],xlims[1])
output_basename = output_dir + '/figures/pole_histogram'
ip.save_png_pdf(output_basename)



# %% md

# ==============================================================================
# ## Calculate edginess
# ==============================================================================

# %% codecell

# For each point
edginess = []
for i, sp in tqdm(spot_props.iterrows()):
    # Get cell id
    cid = sp.cell_id
    if cid > 0:
        cell = cell_dict[cid]
        # print(cell)
        # Get orientation of cell θ
        # Get length of long axis
        # Get length of small axis
        # Get centroid of cell
        # Get spot coords
        sp_coords = np.array([sp['centroid-0'], sp['centroid-1']])
        # print(sp_coords)
        # Calculate coords of cell centerline endpoints:
        # Subtract width from length then divide by 2, get h
        h = (cell['len'] - cell['wid']) / 2
        # print(h)
        # x=±hcos(θ) y=±hsin(θ), e1 and e2
        ori = cell['ori']
        # ori = 1
        sgns = [1,-1]
        es = [
                np.array([sgn*h*math.cos(ori), sgn*h*math.sin(ori)])
                for sgn in sgns
                ]
        # print(es)
        # Decide how to measure spot distance to line:
        # Subtract centroid from spot coords, s
        s = sp_coords - cell['cent']
        # s = np.array([10,5])
        # print(s)
        # fig, ax = ip.general_plot()
        # ax.plot([es[0][1], es[1][1]], [es[0][0], es[1][0]])
        # ax.invert_yaxis()
        # ax.set_aspect('equal')
        # ax.plot(s[1],s[0],'.r', ms=5)
        # Calculate distance of s from both e1 and e2 pick the smallest, return index
        d_es = [np.linalg.norm(s-e) for e in es]
        i_d = np.argmin(d_es)
        # return min distance
        d = np.min(d_es)
        # print('d ', d)
        # # adjust theta depending on which end we chose
        # adj_theta = [0, math.pi][i_d]
        # theta = ori + adj_theta
        # print('theta ',theta)
        # Pick e depending on which end we chose
        e = es[i_d]
        # print('e ', e)
        # Subtract xe from xs, xse
        se = s - e
        # print(se)
        # Get angle between cell and se
        # theta_se = math.atan(xyse[1] / xyse[0])
        # print(theta_se, theta)
        ese_dot = np.dot(e, se)
        # print(ese_dot)
        # If |θ - θse| < π/2
        if ese_dot > 0:
            # Return distance, D=d
            D = d
            # print('if')
            # print(D)
        # Else
        else:
            # θs = tan^-1(ys/xs)
            theta_s = math.atan(s[1] / s[0])
            # hs = sqrt(xs^2 + ys^2)
            hs = math.sqrt(s[0]**2 + s[1]**2)
            # Return distance, D = hs*sin(|θ - θs|)
            D = abs(hs*math.sin(ori - theta_s))
            # print('else')
            # print(D)
        # print(D/h)
        edginess.append(D / h)

# %% codecell
# %% codecell
dims=(1.5,1)
ft=6
xlims=(0,1.5)
bin_size = 0.1
bins = np.arange(0,np.max(edginess),bin_size)
fig, ax = ip.general_plot(dims=dims, ft=ft)
n, bins, patches = ax.hist(edginess, bins=bins, color=(1,0,1))
ax.set_xticks(ticks=[0,0.5,1, 1.5], labels=[])
ax.set_yticks(ticks=[0,2500, 5000], labels=[])
ax.set_xlim(xlims[0],xlims[1])
output_basename = output_dir + '/figures/edge_histogram'
ip.save_png_pdf(output_basename)
print(edginess)




# %% md

# ==============================================================================
# ## Cell spot density plot
# ==============================================================================

# %% codecell
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# um per pixel
resolution = 0.03

# average cell width and length
len_avg = cell_props.major_axis_length.mean() * resolution
wid_avg = cell_props.minor_axis_length.mean() * resolution
lw_ratio = wid_avg / len_avg

norm_loc = []
for i, sp in tqdm(spot_props.iterrows()):
    # Get cell id
    cid = sp.cell_id
    if cid > 0:
        cell = cell_dict[cid]
        # print(cell)
        # Get orientation of cell θ
        # Get length of long axis
        # Get length of small axis
        # Get centroid of cell
        # Get spot coords
        sp_coords = np.array([sp['centroid-0'], sp['centroid-1']])
        # print(sp_coords)
        # print(cell['cent'])
        ori = cell['ori']
        # print(ori)
        # print(es)
        # Decide how to measure spot distance to line:
        # Subtract centroid from spot coords, s
        s = sp_coords - cell['cent']
        # s = np.array([1,0])
        # print(s)
        # Angle of spot to origin
        theta_s = angle_between(np.array([1,0]), s)
        # Adjust angle so that its sign is correct
        sgn = np.sign(s[1])
        sgn = 1 if sgn == 0 else sgn
        theta_s *= sgn
        # print(theta_s)
        # Distance from spot to origin
        hs = math.sqrt(s[0]**2 + s[1]**2)
        # Adjust angles so that cell faces straight down in imshow
        theta_s_adj = theta_s - ori
        # print(theta_s_adj)
        # Get new spot coords after angle adjustment
        r_s = hs * np.cos(theta_s_adj)
        c_s = hs * np.sin(theta_s_adj)
        # print(r_s, c_s)
        # Scale the cell so that length:width is 3:1 and adjust spot coords
        r_s_adj = r_s / (cell['len']) * len_avg
        c_s_adj = c_s / (cell['wid']) * wid_avg
        # Save
        # print(D/h)
        # norm_loc.append([r_s, c_s])
        norm_loc.append([r_s_adj, c_s_adj])


norm_loc_arr = np.array(norm_loc)

# %% codecell
# Plot spots
dims=(10,10)
ft=12
size=15
alpha=0.1
color='k'

norm_loc_arr[::10,0].shape
fig, ax = ip.general_plot(dims=dims, ft=ft)
ax.scatter(norm_loc_arr[:,1], norm_loc_arr[:,0], s=size, alpha=alpha, fc=color, ec=None)
ax.set_aspect('equal')
theta = -np.pi / 2
np.cos(theta), np.sin(theta)

# %% codecell
# Density plot
radius=0.1
area=np.pi*radius**2
nbrs = NearestNeighbors(radius=radius).fit(norm_loc_arr)

# %% codecell
pix_per_um = 500

r_min, r_max = np.min(norm_loc_arr[:,0]), np.max(norm_loc_arr[:,0])
c_min, c_max = np.min(norm_loc_arr[:,1]), np.max(norm_loc_arr[:,1])

r_dim = int((r_max - r_min) * pix_per_um)
c_dim = int((c_max - c_min) * pix_per_um)

r = np.linspace(r_min, r_max, r_dim)
c = np.linspace(c_min, c_max, c_dim)
cc, rr = np.meshgrid(c, r)
rc_grid = np.dstack([rr,cc])
rc_coords = rc_grid.reshape(-1,2)
dists, inds = nbrs.radius_neighbors(rc_coords)
counts = np.array([d.shape[0] for d in dists])
densities = counts / area
densities_grid = densities.reshape(r_dim,c_dim)
plt.hist(densities)

# %% codecell
cmap='inferno'
im_inches=10
fig, ax, cbar = ip.plot_image(
        densities_grid.T,
        cmap=cmap,
        im_inches=im_inches,
        scalebar_resolution=1/pix_per_um
        )

# # Save figure
# plt.figure(fig)
# out_dir = output_dir + '/figures/subcell_spot_density'
# if not os.path.exists(out_dir): os.makedirs(out_dir)
# output_basename = out_dir + '/' + sn + '_subcell_spot_density'
# ip.save_png_pdf(output_basename, bbox_inches=False)
# # Save cbar
# plt.figure(cbar[0])
# output_basename = out_dir + '/' + sn + '_subcell_spot_density_cbar'
# ip.save_png_pdf(output_basename, bbox_inches=False)



# %% codecell
cmap='inferno'
threshold=30000
densities_grid_thresh = densities_grid.copy()
densities_grid_thresh[densities_grid < threshold] = np.nan
cmap_adj = plt.get_cmap(cmap).copy()
cmap_adj.set_bad('k',1.)
fig, ax, cbar = ip.plot_image(
        densities_grid_thresh.T,
        cmap=cmap_adj,
        im_inches=im_inches,
        scalebar_resolution=1/pix_per_um
        )
cbar[1].set_ticks(ticks=[30000,35000,40000])
# Save figure
plt.figure(fig)
out_dir = output_dir + '/figures/subcell_spot_density'
if not os.path.exists(out_dir): os.makedirs(out_dir)
output_basename = out_dir + '/' + sn + '_subcell_spot_density_thresh'
ip.save_png_pdf(output_basename, bbox_inches=False)
# Save cbar
plt.figure(cbar[0])
output_basename = out_dir + '/' + sn + '_subcell_spot_density_thresh_cbar'
ip.save_png_pdf(output_basename, bbox_inches=False)


# %% codecell
from matplotlib import colors
cmap='inferno'
im_inches=10
densities_grid_log = densities_grid
fig, ax, cbar = ip.plot_image(
        densities_grid_log.T,
        cmap=cmap,
        im_inches=im_inches,
        scalebar_resolution=1/pix_per_um,
        norm=colors.PowerNorm(2.5)
        )
cbar[1].set_ticks(ticks=[0, 20000,30000,40000])
# # Save figure
plt.figure(fig)
out_dir = output_dir + '/figures/subcell_spot_density'
if not os.path.exists(out_dir): os.makedirs(out_dir)
output_basename = out_dir + '/' + sn + '_subcell_spot_density_powernorm2_5'
ip.save_png_pdf(output_basename, bbox_inches=False)
# Save cbar
plt.figure(cbar[0])
output_basename = out_dir + '/' + sn + '_subcell_spot_density_powernorm2_5_cbar'
ip.save_png_pdf(output_basename, bbox_inches=False)


# %% codecell
