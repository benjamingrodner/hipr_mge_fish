# %% md

# # Figure 3c: Two person experiment. Generate background mask.

# Used "hiprfish_imaging_py38" conda environment

# %% md

# =============================================================================
# ## Setup
# =============================================================================

# %% md

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
from skimage.restoration import richardson_lucy
from scipy.stats import linregress
from sklearn.cluster import MiniBatchKMeans
from time import time
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from tqdm import tqdm
from PIL import Image
from collections import defaultdict



# from tqdm import tqdm
gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/workdir/bmg224/manuscripts/mgefish/code/fig_4/genomic_amr_genes/segmentation'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

# Go into your configuration file and adjust all of the 'Paths' so they are relative to the workdir.

# Also adjust the 'Inputs' parameters to fit the images.

# %% codecell
config_fn = 'config_mgefish.yaml' # relative path to config file from workdir

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

# =============================================================================
# ## Convert czi files to numpy arrays
# =============================================================================

# Get the input filenames. Reload the config file if you need to update the regular expression.

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

input_filenames = glob.glob(config['input_dir'] + '/' + config['input_regex']
                            + config['input_ext'])
input_filenames.sort()
print(len(input_filenames))
input_filenames

# %% md

# To run all the samples write a full input table.

# %% codecell
input_fns_split = [os.path.split(fn)[1] for fn in input_filenames]
sample_names = [re.sub(config['input_ext'], '', fn) for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['input_table_fn'], index=False)
input_table.values

# %% md

# Use the snakemake rule to do the conversion

# Write the snakemake execution code to a bash script.

# %% codecell
dry_run = False  # Just create DAG if True
n_cores = 1  # number of allowed cores for the snakemake to use
force_run = False  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['snakefile_convert']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
command = " ".join(['snakemake', '-s', snakefile, '--configfile', config_fn, '-j',
                    str(n_cores), dr, fr])

run_fn = 'run_{}.sh'.format(snakefile)
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

# %% md

# =============================================================================
# ## Choose background threshold values
# =============================================================================

# Make a folder for background thresholding stuff

# %% codecell
# Get ouput format
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

spot_mask_bg_rough_fmt = config['output_dir'] + '/' + config['spot_mask_bg_rough_fmt']
for sn in sample_names:
    fn = spot_mask_bg_rough_fmt.format(sample_name=sn, spot_chan='')
    # split by dirs
    normalized_path = os.path.normpath(fn)
    path_components = normalized_path.split(os.sep)
    dir = '.'
    for d in path_components[:-1]:
        dir += '/' + d
        if not os.path.exists(dir):
            # If dir doesnt exist, make dir
            os.makedirs(dir)
            print('Made dir: ', dir)

# %% md

# Get background values

# %% codecell
bgs=5
nc=5
tnc=4
bgt=0.0015

spot_mask_bg_rough_fmt = config['output_dir'] + '/' + config['spot_mask_bg_rough_fmt']
# spot_vals_bgrough_fmt = config['output_dir'] + '/' + config['spot_vals_bgrough_fmt']
bg_df = pd.DataFrame([])
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch in config['spot_seg']['channels']:
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        # Get rough background mask
        # bg_mask = sf.get_background_mask(
        #             raw_ch, bg_smoothing=bgs, n_clust_bg=nc, top_n_clust_bg=tnc
        #             )
        bg_mask = sf.get_background_mask(
                    raw_ch, bg_threshold=bgt
                    )
        # Save rough background mask
        output_filename = spot_mask_bg_rough_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, bg_mask)
        print(output_filename)
        # Get the mean value of background
        bg = raw_ch[bg_mask == 0]
        mean = np.mean(bg)
        print(mean)
        std = np.std(bg)
        bg_df[sn + '_spotchan_' + str(s_ch)] = [mean, std]
# Save background values
# output_filename = spot_vals_bgrough_fmt.format(sample_name=sn, spot_chan=s_ch)
# bg_df.to_csv(output_filename)
# print(output_filename)


# %% md

# Plot background values and images with background masked

# %% codecell
clims = (0,0.01)
cmap = ip.get_cmap('inferno')(np.linspace(0,1,256))
cmap[0] = [0.5,0.5,0.5,1]
cmap = ip.ListedColormap(cmap)
dims=(2,2)
lw=1
ft=6

# bg_df = ip.load_output_file(config, 'spot_vals_bgrough_fmt')
fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
ticks = []
for k, sn in enumerate(sample_names):
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    shift = k
    ticks.append(shift)
    for l, s_ch in enumerate(config['spot_seg']['channels']):
        mean, std = bg_df[sn + '_spotchan_' + str(s_ch)].values
        shift2 = shift + l*0.2
        ax.errorbar(shift2, mean, yerr=std, marker='.')
        print(sn)
        raw_ch = raw[:,:,s_ch]
        bg_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=s_ch)
        raw_mask = raw_ch * bg_mask
        im_list = [raw_ch, raw_mask]
        fig2, ax2, cbar = ip.subplot_square_images(im_list, (1,2), clims=[clims,clims], cmaps=[cmap, cmap])
        plt.figure(fig2)
plt.figure(fig)
ax.set_xticks(ticks)
ax.set_xticklabels([])
plt.show()

# %% md

# Get threshold curves

# %% codecell
blims = (0,20)
n_bthr = 20
bthreshs = np.linspace(blims[0], blims[1], n_bthr)

# spot_vals_bg_curve_fmt = config['output_dir'] + '/' + config['spot_vals_bg_curve_fmt']
b_curves_df = pd.DataFrame([])
for sn in tqdm(sample_names):
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    # Get proper channel
    for s_ch in config['spot_seg']['channels']:
        raw_ch = raw[:,:,s_ch]
        # Get background mask
        # bg_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=s_ch)
        # Get background valeue
        bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
        # Get SNR image
        raw_snr = raw_ch / bg_val
        # Get the number of pixels before thresholding
        n = raw_snr.ravel().shape[0]
        # Calculate ratio and add to curve values
        curve = [raw_snr[raw_snr > thr].shape[0] / n for thr in bthreshs]
        b_curves_df[sn + '_spotchan_' + str(s_ch)] = curve
        # Save threshold curve values
# output_filename = spot_vals_bg_curve_fmt
# curves_df.to_csv(output_filename, index=False)
# curves_df.shape

# %% md

# Plot curves

# %% codecell
# Pick a threshold
# thr_pk_snr = [40]
thresh_pick_snr = [3,5,3]
xlims=(1,20)
ylims=[0,0.25]
dims=(4,4)
colors = ip.get_cmap('tab10').colors
cols = colors[:3]


# spot_curve_bg_fmt = config['output_dir'] + '/{sample_name}_chan_{spot_chan}_curve_bg'
# curves_df = ip.load_output_file(config, 'spot_vals_bg_curve_fmt')
# Start a figure
for s_ch, thr_pk_snr, clr in zip(config['spot_seg']['channels'], thresh_pick_snr, cols):
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    for sn in sample_names:
        # for sn, thr_pk_snr in zip(sample_names,thresh_pick_snr):
    # print(sn)
        # Get curve
        curve = b_curves_df[sn + '_spotchan_' + str(s_ch)].values
        # Plot curve and threshold
        ax.plot(bthreshs, curve, lw=lw, color=clr)
        ax.plot([thr_pk_snr]*2, [0,1], color=clr, lw=lw)
        ax.plot(list(xlims), [0,0],'k',lw=lw*0.5)
        # adjust plot
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    # # save plot
    # output_basename = spot_curve_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
    # ip.save_png_pdf(output_basename)
    # show plot
    plt.show()

# # %% md
#
# # Picked a threshold so that 1% of pixels are left
#
# # %% codecell
# pix_pct = 0.01
#
# thresholds = []
# for sn in sample_names:
# # for sn, thr_pk_snr in zip(sample_names,thresh_pick_snr):
#     print(sn)
#     for s_ch in config['spot_seg']['channels']:
#         # Get curve
#         curve = curves_df[sn + '_spotchan_' + str(s_ch)].values
#         curve = np.flip(curve)
#         # Get the x values
#         xv = np.flip(bthreshs)
#         # Get the threshold value
#         thr = np.interp(pix_pct, curve, xv)
#         print(thr)
#         thresholds.append(thr)

# %% md

# Pick only one image

# %% codecell
sample_names = [sample_names[0]]


# %% md

# Plot ordered pixel intensities

# %% codecell
bg_thr = 0.001
thresh_pick = [0.002,0.0065,0.0025]


skip = 100
dims=(5,5)
ft=12
lw=1
s=1
lw=1
ylims=(0,0.01)
for s_ch, thr_pk in zip(config['spot_seg']['channels'], thresh_pick):
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    print('channel: ', s_ch)
    for sn in sample_names:
        raw = ip.load_output_file(config, 'raw_fmt', sn)
        raw_ints = raw[:,:,s_ch].ravel()
        raw_ints_bg = raw_ints * (raw_ints > bg_thr)
        raw_ints_sort = np.sort(raw_ints_bg)
        count = raw_ints_sort.shape
        inds = np.arange(0,raw_ints_sort.shape[0], skip)
        raw_ints_sub = raw_ints_sort[inds]
        x = np.arange(raw_ints_sub.shape[0])
        ax.scatter(x, raw_ints_sub, s=s)
        ax.plot([0,np.max(x)], [thr_pk]*2, color='k', lw=lw)
    ax.set_ylim(ylims[0], ylims[1])
    plt.show()
    plt.close()



# %% md

# Show pre-thresholded images vs post

# %% codecell
# Show full FOV
clims = [(0,0.0025),(0,0.0075), (0,0.003)]

im_inches = 10

spot_mask_bg_fmt = config['output_dir'] + '/' + config['spot_mask_bg_fmt']
# for sn, thr_pk_snr in zip(sample_names, thresholds):
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    print(sn)
    # for s_ch in config['spot_seg']['channels']:
    # for s_ch, thr_pk_snr in zip(config['spot_seg']['channels'], thresh_pick_snr):
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
        print(s_ch)
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        # Get background valeue
        # bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
        # Get SNR image
        # raw_snr = raw_ch / bg_val
        # Get mask
        mask = raw_ch > thr_pk
        # mask = raw_snr > thr_pk_snr
        # print(raw_snr[mask].shape[0])
        # Label regions in mask
        mask_l = sf.label(mask)
        # Save mask
        output_filename = spot_mask_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, mask_l)
        # Mask image
        raw_mask = raw_ch * mask
        # raw_mask = raw_snr * mask
        # show raw image next to thresholded image
        im_list = [raw_ch, raw_mask, mask]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,''], im_inches=im_inches)
        plt.show()
        plt.close()

# %% codecell
# show zoom image
zc = [5000, 0]
zs = [3000, 3000]
clims = [(0,0.0025),(0,0.0075), (0,0.003)]

im_inches = 10

spot_mask_bg_fmt = config['output_dir'] + '/' + config['spot_mask_bg_fmt']
# for sn, thr_pk_snr in zip(sample_names, thresholds):
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    print(sn)
    # for s_ch in config['spot_seg']['channels']:
    # for s_ch, thr_pk_snr in zip(config['spot_seg']['channels'], thresh_pick_snr):
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
        print(s_ch)
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        # Get background valeue
        # bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
        # Get SNR image
        # raw_snr = raw_ch / bg_val
        # Get mask
        mask = raw_ch > thr_pk
        # mask = raw_snr > thr_pk_snr
        # print(raw_snr[mask].shape[0])
        # Label regions in mask
        mask_l = sf.label(mask)
        # Save mask
        output_filename = spot_mask_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, mask_l)
        # Mask image
        raw_mask = raw_ch * mask
        # raw_mask = raw_snr * mask
        # show raw image next to thresholded image
        im_list = [raw_ch, raw_mask, mask]
        im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,''], im_inches=im_inches)
        plt.show()
        plt.close()



# %% md

# =============================================================================
# ## Filter out debris
# =============================================================================

# Get region props from thresholded image
# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% codecell
spot_props_bg_fmt = config['output_dir'] + '/' + config['spot_props_bg_fmt']
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch in config['spot_seg']['channels']:
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        # Get mask
        mask = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=s_ch)
        # Get regionprops in mask
        props = sf.measure_regionprops(mask, raw_ch)
        # Save regionprops
        output_filename = spot_props_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
        props.to_csv(output_filename, index=False)
        print(output_filename)


# # %% md
#
# # Get area threshold curves
#
# # %% codecell
# alims = (0,20000)
# n_athr = 5000
# athreshs = np.linspace(alims[0], alims[1], n_athr)
# # spot_vals_debris_curve_fmt: spot_mask/fig_3c_spot_vals_bgrough.csv
# # spot_vals_debris_curve_fmt = config['output_dir'] + '/' + config['spot_vals_debris_curve_fmt']
# s_curves_df = pd.DataFrame([])
# for sn in sample_names:
#     for s_ch in config['spot_seg']['channels']:
#         # Get regionprops
#         props = ip.load_output_file(config, 'spot_props_bg_fmt', sn, spot_chan=s_ch)
#         # Get curve value at threshold
#         n = props.shape[0]
#         curve = [props[props['area'] < thr].shape[0] / n for thr in athreshs]
#         s_curves_df[sn + '_spotchan_' + str(s_ch)] = curve
# # Save curves
# # output_filename = spot_vals_debris_curve_fmt
# # curves_df.to_csv(output_filename, index=False)
# curves_df.iloc[-1,:]
#
# # %% md
#
# # Plot curves
#
# # There really arent any extra large debris areas and we can't pick an area threshold without removing real spots. Instead we will do area threasholding after the segmentation.
#
# # So we will allow all objects through the debris filter
#
# # %% codecell
# thr_pk_area = [5000]  # Maximum object size (in pixels)
#
# xlims=(0,20000)
# ylims=[0.975,1.01]
# dims=(10,10)
# h=1
# cols = colors[:3]
#
# # spot_curve_debris_fmt = config['output_dir'] + '/fig_3c_curve_debris'
# # spot_vals_debris_curve_fmt = config['output_dir'] + '/' + config['spot_vals_debris_curve_fmt']
# # curves_df = ip.load_output_file(config, 'spot_vals_debris_curve_fmt')
# # Start a figure
# fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
# for sn in sample_names:
#     for s_ch, clr, tpa in zip(config['spot_seg']['channels'], cols, thr_pk_area):
#         # Get curve
#         curve = s_curves_df[sn + '_spotchan_' + str(s_ch)].values
#         # Plot curve and threshold
#         ax.plot(athreshs, curve, lw=lw, color=clr)
#         ax.plot([tpa]*2, [0,h], color=clr, lw=lw)
# ax.plot(list(xlims), [h,h],'k',lw=lw*0.5)
# # adjust plot
# ax.set_xlim(xlims)
# ax.set_ylim(ylims)
# # save plot
# # output_basename = spot_curve_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
# # ip.save_png_pdf(output_basename)
# # show plot
# plt.show()


# %% md

# Get an elbow plot of spot size

# # %% codecell
# props_area = []
# for sn in sample_names:
#     pas = []
#     for s_ch in config['spot_seg']['channels']:
#         # Get regionprops
#         props = ip.load_output_file(config, 'spot_props_bg_fmt', sn, spot_chan=s_ch)
#         pas.append(props['area'].sort_values())
#         # Get curve
#         # Plot curve and threshold
#     props_area.append(pas)

# %% codecell
thr_pk_area = [1250, 1250, 1250]
ylims=(0,2000)
lw=1
s=1
for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
    print(s_ch)
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    for sn in sample_names:
        print(sn)
        props = ip.load_output_file(config, 'spot_props_bg_fmt', sn, spot_chan=s_ch)
        pa = props['area'].sort_values()
        ax.scatter(np.arange(pa.shape[0]), pa, s=s)
        xlims = ax.get_xlim()
# for tpa in thr_pk_area:
    ax.plot(xlims, [tpa]*2, color='k', lw=lw*2)
    ax.set_ylim(ylims[0],ylims[1])
    plt.show()
    plt.close()


# %% md

# Save area filtered mask

# %% codecell
spot_mask_bg_debris_fmt = config['output_dir'] + '/' + config['spot_mask_bg_debris_fmt']
spot_props_bg_debris_fmt = config['output_dir'] + '/' + config['spot_props_bg_debris_fmt']
for sn in sample_names:
    # Load raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
    # for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
        # pick proper channel
        raw_ch = raw[:,:,s_ch]
        # Load mask
        mask_old = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=s_ch)
        # load regionprops
        props = ip.load_output_file(config, 'spot_props_bg_fmt', sn, spot_chan=s_ch)
        # Mask raw image
        # Get bool for area threshold
        props['area_thresh'] = (props.area < tpa).astype(int)
        # props['area_thresh'] = (props.area < tpa).astype(int)
        # Save new regionprops
        output_filename = spot_props_bg_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
        props.to_csv(output_filename, index=False)
        # Remove filtered objects from mask
        mask_new = ip.filter_seg_objects(mask_old, props, 'area_thresh')
        print(props.shape[0], props.area_thresh.sum())
        # New mask on raw image
        # Save new mask
        output_filename = spot_mask_bg_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, mask_new)
        print(output_filename)
        # show the images side by side

# %% md

# Show filtering. From left to right: raw image, SNR thresholded mask, area thresholded mask

# %% codecell
# clims = (0,0.01)
im_inches=10

for sn in sample_names:
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
    # for s_ch in config['spot_seg']['channels']:
        raw_ch = raw[:,:,s_ch]
        mask_old = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=s_ch)
        raw_mask_old = raw_ch * (mask_old > 0)
        mask_new = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
        raw_mask_new = raw_ch * (mask_new > 0)
        im_list = [raw_ch, mask_old>0, mask_new>0]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,(),()], im_inches=im_inches)

# %% codecell
# clims = (0,0.01)
zc = [5000, 0]
zs = [3000, 3000]

for sn in sample_names:
    print(sn)
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    # for s_ch in config['spot_seg']['channels']:
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
        print(s_ch)
        raw_ch = raw[:,:,s_ch]
        mask_old = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=s_ch)
        raw_mask_old = raw_ch * (mask_old > 0)
        mask_new = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
        raw_mask_new = raw_ch * (mask_new > 0)
        im_list = [raw_ch, mask_old>0, mask_new>0]
        im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
        plt.show()
        plt.close()


# %% md

# =============================================================================
# ## Second round of intensity filtering
# =============================================================================

# %% codecell
thr_pk_int = [0.0035, 0.02, 0.01]
ylims=(0,0.04)
dims=(5,5)
lw=1
s=1
ft=12
for s_ch, tpi in zip(config['spot_seg']['channels'], thr_pk_int):
    print(s_ch)
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    for sn in sample_names:
        print(sn)
        props = ip.load_output_file(config, 'spot_props_bg_debris_fmt', sn, spot_chan=s_ch)
        pa = props['max_intensity'].sort_values(ascending=False)
        ax.scatter(np.arange(pa.shape[0]), pa, s=s)
        xlims = ax.get_xlim()
# for tpa in thr_pk_area:
    ax.plot(xlims, [tpi]*2, color='k', lw=lw*2)
    ax.set_ylim(ylims[0],ylims[1])
    plt.show()
    plt.close()


# %% codecell
spot_mask_bg_int_fmt = config['output_dir'] + '/' + config['spot_mask_bg_int_fmt']
spot_props_bg_int_fmt = config['output_dir'] + '/' + config['spot_props_bg_int_fmt']

sn = sample_names[0]
raw = ip.load_output_file(config, 'raw_fmt', sn)
for s_ch, tpi in zip(config['spot_seg']['channels'], thr_pk_int):
# for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
    # pick proper channel
    raw_ch = raw[:,:,s_ch]
    # Load mask
    mask_old = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
    # load regionprops
    props = ip.load_output_file(config, 'spot_props_bg_debris_fmt', sn, spot_chan=s_ch)
    # Mask raw image
    # Get bool for area threshold
    props['int_thresh'] = (props.max_intensity > tpi).astype(int)
    # props['area_thresh'] = (props.area < tpa).astype(int)
    # Save new regionprops
    output_filename = spot_props_bg_int_fmt.format(sample_name=sn, spot_chan=s_ch)
    props.to_csv(output_filename, index=False)
    # Remove filtered objects from mask
    mask_new = ip.filter_seg_objects(mask_old, props, 'int_thresh')
    print(np.unique(mask_old).shape[0], np.unique(mask_new).shape[0])
    # New mask on raw image
    # Save new mask
    output_filename = spot_mask_bg_int_fmt.format(sample_name=sn, spot_chan=s_ch)
    np.save(output_filename, mask_new)
    print(output_filename)
    # show the images side by side

# %% codecell


print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_old = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
    raw_mask_old = raw_ch * (mask_old > 0)
    mask_new = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    im_list = [raw_ch, mask_old>0, mask_new>0]
    # im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
    plt.show()
    plt.close()


# %% codecell
# clims = (0,0.01)
zc = [4000, 0]
zs = [3000, 3000]

print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_old = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
    raw_mask_old = raw_ch * (mask_old > 0)
    mask_new = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    im_list = [raw_ch, mask_old>0, mask_new>0]
    im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
    plt.show()
    plt.close()


# %% md

# =============================================================================
# ## Eccentricity filtering
# =============================================================================

# %% codecell
thr_pk_ecc = [0.99, 0.99, 0.99]
ylims=(0,0.04)
dims=(5,5)
lw=1
s=1
for s_ch, tpe in zip(config['spot_seg']['channels'], thr_pk_ecc):
    print(s_ch)
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    for sn in sample_names:
        print(sn)
        props = ip.load_output_file(config, 'spot_props_bg_int_fmt', sn, spot_chan=s_ch)
        pa = props['eccentricity'].sort_values()
        ax.scatter(np.arange(pa.shape[0]), pa, s=s)
        xlims = ax.get_xlim()
# for tpa in thr_pk_area:
    ax.plot(xlims, [tpe]*2, color='k', lw=lw*2)
    # ax.set_ylim(ylims[0],ylims[1])
    plt.show()
    plt.close()


# %% codecell
spot_mask_bg_ecc_fmt = config['output_dir'] + '/' + config['spot_mask_bg_ecc_fmt']
spot_props_bg_ecc_fmt = config['output_dir'] + '/' + config['spot_props_bg_ecc_fmt']

sn = sample_names[0]
raw = ip.load_output_file(config, 'raw_fmt', sn)
for s_ch, tpe in zip(config['spot_seg']['channels'], thr_pk_ecc):
# for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
    # pick proper channel
    raw_ch = raw[:,:,s_ch]
    # Load mask
    mask_old = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    # load regionprops
    props = ip.load_output_file(config, 'spot_props_bg_int_fmt', sn, spot_chan=s_ch)
    # Mask raw image
    # Get bool for area threshold
    props['ecc_thresh'] = (props.eccentricity < tpe).astype(int)
    # props['area_thresh'] = (props.area < tpa).astype(int)
    # Save new regionprops
    output_filename = spot_props_bg_ecc_fmt.format(sample_name=sn, spot_chan=s_ch)
    props.to_csv(output_filename, index=False)
    # Remove filtered objects from mask
    mask_new = ip.filter_seg_objects(mask_old, props, 'ecc_thresh')
    print(np.unique(mask_old).shape[0], np.unique(mask_new).shape[0])

    # New mask on raw image
    # Save new mask
    output_filename = spot_mask_bg_ecc_fmt.format(sample_name=sn, spot_chan=s_ch)
    np.save(output_filename, mask_new)
    print(output_filename)
    # show the images side by side

# %% codecell
im_inches=10

print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_old = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_mask_old = raw_ch * (mask_old > 0)
    mask_new = ip.load_output_file(config, 'spot_mask_bg_ecc_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    im_list = [raw_ch, mask_old>0, mask_new>0]
    # im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
    plt.show()
    plt.close()

# %% codecell
# clims = (0,0.01)
zc = [4000, 3000]
zs = [3000, 3000]

print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_old = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_mask_old = raw_ch * (mask_old > 0)
    mask_new = ip.load_output_file(config, 'spot_mask_bg_ecc_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    im_list = [raw_ch, mask_old>0, mask_new>0]
    im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
    plt.show()
    plt.close()

# %% md

# =============================================================================
# ## small obj filtering ?
# =============================================================================

# %% codecell
thr_pk_area_sm = [10, 10, 60]
ylims=(0,100)
dims=(5,5)
lw=1
s=1
for s_ch, tpas in zip(config['spot_seg']['channels'], thr_pk_area_sm):
    print(s_ch)
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    for sn in sample_names:
        print(sn)
        props = ip.load_output_file(config, 'spot_props_bg_ecc_fmt', sn, spot_chan=s_ch)
        props.columns
        bools = (props.area_thresh * props.int_thresh * props.ecc_thresh).astype(bool)
        pa = props.loc[bools, 'area'].sort_values()
        ax.scatter(np.arange(pa.shape[0]), pa, s=s)
        xlims = ax.get_xlim()
# for tpa in thr_pk_area:
    ax.plot(xlims, [tpas]*2, color='k', lw=lw*2)
    ax.set_ylim(ylims[0],ylims[1])
    plt.show()
    plt.close()

#
# # %% codecell
# spot_mask_bg_area_sm_fmt = config['output_dir'] + '/' + config['spot_mask_bg_area_sm_fmt']
# spot_props_bg_area_sm_fmt = config['output_dir'] + '/' + config['spot_props_bg_area_sm_fmt']
#
# sn = sample_names[0]
# raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch, tpas in zip(config['spot_seg']['channels'], thr_pk_area_sm):
# # for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
#     # pick proper channel
#     raw_ch = raw[:,:,s_ch]
#     # Load mask
#     mask_old = ip.load_output_file(config, 'spot_mask_bg_ecc_fmt', sn, spot_chan=s_ch)
#     # load regionprops
#     props = ip.load_output_file(config, 'spot_props_bg_ecc_fmt', sn, spot_chan=s_ch)
#     # Mask raw image
#     # Get bool for area threshold
#     props['area_sm_thresh'] = (props.area > tpas).astype(int)
#     # props['area_thresh'] = (props.area < tpa).astype(int)
#     # Save new regionprops
#     output_filename = spot_props_bg_area_sm_fmt.format(sample_name=sn, spot_chan=s_ch)
#     props.to_csv(output_filename, index=False)
#     # Remove filtered objects from mask
#     mask_new = ip.filter_seg_objects(mask_old, props, 'area_sm_thresh')
#     print(np.unique(mask_old).shape[0], np.unique(mask_new).shape[0])
#
#     # New mask on raw image
#     # Save new mask
#     output_filename = spot_mask_bg_area_sm_fmt.format(sample_name=sn, spot_chan=s_ch)
#     np.save(output_filename, mask_new)
#     print(output_filename)
#     # show the images side by side
#
# # %% codecell
# # clims = (0,0.01)
# zc = [5500, 1500]
# zs = [500, 500]
#
# print(sn)
# raw = ip.load_output_file(config, 'raw_fmt', sn)
# # for s_ch in config['spot_seg']['channels']:
# for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
#     print(s_ch)
#     raw_ch = raw[:,:,s_ch]
#     mask_old = ip.load_output_file(config, 'spot_mask_bg_ecc_fmt', sn, spot_chan=s_ch)
#     raw_mask_old = raw_ch * (mask_old > 0)
#     mask_new = ip.load_output_file(config, 'spot_mask_bg_area_sm_fmt', sn, spot_chan=s_ch)
#     raw_mask_new = raw_ch * (mask_new > 0)
#     im_list = [raw_ch, mask_old>0, mask_new>0]
#     im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
#     ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
#     plt.show()
#     plt.close()

# # %% md
#
# # =============================================================================
# # ## Colocalization
# # =============================================================================

# %% codecell

r_um = 0.3

r_pix = r_um / config['resolution']
neighs = {}
ch_coords = []
# Load props
for s_ch in config['spot_seg']['channels']:
    print(s_ch)
    props = ip.load_output_file(config, 'spot_props_bg_ecc_fmt', sn, spot_chan=s_ch)
    # Create kneighbors for each channel
    props.columns
    bools = (props.area_thresh.values * props.int_thresh.values * props.ecc_thresh.values) == 1
    props_ = props[bools]
    coords = [list(eval(c)) for c in props_.centroid.values]
    print(len(coords))
    neighs[s_ch] = NearestNeighbors(radius=r_pix).fit(coords)
    ch_coords.append(coords)

# %% codecell
# Get kneighbors in certain radius for both other channel
dict_frac_assoc = {}
for s_ch, coords in zip(config['spot_seg']['channels'], ch_coords):
    dict_frac_assoc[s_ch] = {}
    for s_ch_2 in config['spot_seg']['channels']:
        if not s_ch == s_ch_2:
            dists, inds = neighs[s_ch_2].radius_neighbors(coords)

            bool_ch = [i.shape[0]>0 for i in inds]
            count = sum(bool_ch)
            print(len(coords))
            dict_frac_assoc[s_ch][s_ch_2] = count / len(coords)

dict_frac_assoc

# count fraction assoc with each
# count frac assoc with both

# %% codecell
# barplot
ft=6
dims=(0.3, 1.05)
lw=1
line_col='k'
fc='k'
height=0.5
barlw=0.5
col_dict = {0:(1,1,0), 1:(0,1,1), 3:(1,0,1)}
ylims=[-1,2]
xlims=[-1,1.5]
col='k'

for s_ch, coords_ch in zip(config['spot_seg']['channels'], ch_coords):
    print(s_ch)
    fig, ax = ip.general_plot(ft=ft, dims=dims, lw=lw, col=line_col)
    data = dict_frac_assoc[s_ch]
    order = [k for k in data.keys()]
    order.reverse()
    width = [data[k] for k in order]
    y = np.arange(len(order))
    barlist = ax.bar(y, height=width, width=height, fc=fc, lw=barlw)
    # barlist = ax.barh(y, width=width, height=height, ec=line_col, lw=barlw)
    # for b, k in zip(barlist, order):
    #     b.set_facecolor(col_dict[k])
    # ax.set_ylim(ylims[0],ylims[1])
    # ax.set_xlim(xlims[0], xlims[1])
    ax.set_xticks(labels=[],ticks=[])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    # if not s_ch == 3:
    ax.set_yticks(ticks=[0,0.1,0.2,0.3], labels=[])
    out_dir = config['output_dir'] + '/spot_assoc_barplots'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_bn = out_dir + '/' + sn + '_bar_plots_spot_chan_' + str(s_ch)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()



# %% codecell
size=10
alpha=0.5
fig, ax = ip.general_plot(dims = (20,20))
ax.imshow(np.zeros(raw_ch.shape + (3,)))
for s_ch, tpas in zip(config['spot_seg']['channels'], thr_pk_area_sm):
    print(s_ch)
    props = ip.load_output_file(config, 'spot_props_bg_ecc_fmt', sn, spot_chan=s_ch)
    # Create kneighbors for each channel
    props.columns
    bools = (props.area_thresh.values * props.int_thresh.values * props.ecc_thresh.values) == 1
    props_ = props[bools]
    coords = np.array([list(eval(c)) for c in props_.centroid.values])
    print(coords.shape)
    ax.scatter(coords[:,1],coords[:,0], s=size, alpha=alpha, label=s_ch)
ax.legend()





# # %% md
#
# # =============================================================================
# # ## Nearest neighbor plots
# # =============================================================================

# %% codecell
# Get nearest neighbors for each pair

thresh_cols = ['area_thresh','int_thresh','ecc_thresh']
dict_nneighbor_dists = defaultdict(dict)
for ch_0 in config['spot_seg']['channels']:
    props_0 = ip.load_output_file(
            config,
            'spot_props_bg_ecc_fmt',
            sn,
            spot_chan=ch_0
            )
    bools_0_list = [props_0[c].values for c in thresh_cols]
    bools_0 = (np.prod(np.array(bools_0_list), axis=0) == 1)
    # bools_0 = (props_0.area_thresh.values * props_0.int_thresh.values * props_0.ecc_thresh.values) == 1
    props_0_filt = props_0[bools_0]
    print(ch_0)
    print(props_0_filt.shape)
    coords_0 = [list(eval(c)) for c in props_0_filt.centroid.values]
    nbrs = NearestNeighbors(n_neighbors=1).fit(coords_0)
    for ch_1 in config['spot_seg']['channels']:
        if not ch_0 == ch_1:
            print('  ', ch_1)
            props_1 = ip.load_output_file(
                    config,
                    'spot_props_bg_ecc_fmt',
                    sn,
                    spot_chan=ch_1
                    )
            bools_1_list = [props_1[c].values for c in thresh_cols]
            bools_1 = (np.prod(np.array(bools_1_list), axis=0) == 1)
            # bools_1 = (props_1.area_thresh.values * props_1.int_thresh.values * props_1.ecc_thresh.values) == 1
            props_1_filt = props_1[bools_1]
            print('  ', props_1_filt.shape)
            coords_1 = [list(eval(c)) for c in props_1_filt.centroid.values]
            dists, inds = nbrs.kneighbors(coords_1)
            dict_nneighbor_dists[ch_1][ch_0] = dists

# %% codecell
# plot
dims=(5,5)
lw=1
line_col='k'
bin_width_um=0.5
ft=14
alpha=1
xlims=(0,20)
ylims=(0,0.35)
color=(0,0,0)

max=0
for i, j in dict_nneighbor_dists.items():
    for k, l in j.items():
        lmax = np.max(l)
        max = max if max > lmax else lmax

max_um = max * config['resolution']
bins = np.arange(0,max_um, bin_width_um)
dict_ch_col = {0:(1,1,0),1:(0,1,1),3:(1,0,1)}
for ch_0 in config['spot_seg']['channels']:
    print(ch_0)
    for ch_1 in config['spot_seg']['channels']:
        if not ch_0 == ch_1:
            fig, ax = ip.general_plot(dims=dims, lw=lw, col=line_col, ft=ft)
            # color = dict_ch_col[ch_1]
            dists = dict_nneighbor_dists[ch_0][ch_1] * config['resolution']
            hist, bin_edges = np.histogram(dists, bins=bins)
            print(sum(hist))
            hist = hist / np.sum(hist)
            # x = ip.get_line_histogram_x_vals(bin_edges) * config['resolution']
            # ax.plot(x, hist, color=color, lw=lw)
            x = bin_edges[:-1] + 0.5*bin_width_um
            ax.bar(x, hist ,width=bin_width_um, fc=color, alpha=alpha)
            # n, bins, patches = ax.hist(dists, bins=bins,fc=color, ec=line_col, density=True, alpha=alpha)
            # for item in patches:
            #     item.set_height(item.get_height()/sum(x))
            ax.set_xlim(xlims[0],xlims[1])
            ax.set_ylim(ylims[0],ylims[1])
            plt.show()
            plt.close()


# %% md

# ==============================================================================
# ## get correlation between colors
# ==============================================================================

# %% codecell
# clims = (0,0.01)
zc = [3000, 0]
zs = [3000, 3000]

print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
raw_masked = []
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_new = ip.load_output_file(config, 'spot_mask_bg_ecc_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    raw_masked.append(raw_mask_new)

raw_masked = np.dstack(raw_masked)
raw_masked.shape

# %% codecell
# Divide image into regions,
shape = raw_ch.shape
region_size_um = 2
region_size = int(region_size_um / config['resolution'])
rows = np.arange(0,shape[0],region_size)
columns = np.arange(0,shape[1],region_size)
rows.shape, columns.shape


# %% codecell
# get color intensity correlation across all regions
dict_corr = {'0,1':[[],[]], '0,2':[[],[]],'1,2':[[],[]]}
dict_keys_order = ['0,1','0,2','1,2']
bool_selector = np.array([
        [1,1,0],
        [1,0,1],
        [0,1,1]
        ])
for i in tqdm(range(rows.shape[0] - 1)):
    r_0, r_1 = rows[i], rows[i+1]
    r = raw_masked[r_0:r_1,:,:]

    for j in range(columns.shape[0] - 1):
        c_0, c_1 = columns[j], columns[j+1]
        square = r[:, c_0:c_1, :]
        maxs = [np.max(square[:,:,i]) for i in range(raw_masked.shape[2])]
        bools = np.array([m > 0 for m in maxs])
        # bools = np.array([m > t for m, t in zip(maxs, thresh_chans)])

        if sum(bools) == 1:
            selec = bool_selector[bools,:].squeeze().astype(bool)
            keys_towrite = np.array(dict_keys_order)[selec]
        elif sum(bools) > 1:
            keys_towrite = dict_keys_order
        else:
            keys_towrite = []

        for key in keys_towrite:
            inds = list(eval(key))
            dict_corr[key][0].append(maxs[inds[0]])
            dict_corr[key][1].append(maxs[inds[1]])



# %% codecell
# PLot correlations
channel_names=['patA','patB','rRNA','adeF']
dims = [1.,1.]
size = 1
col = 'k'
line_col = 'r'
ft=12

names_spot_chan = [channel_names[i] for i in [0,1,3]]
for k in dict_keys_order:
    inds = list(eval(k))
    xlab, ylab = [names_spot_chan[i] for i in inds]
    x,y = dict_corr[k]
    fig, ax = ip.general_plot(dims=dims, ft=ft)
    ax.scatter(x,y,s=size, c=col)

    # coef = np.polyfit(x,y,1)
    # poly1d_fn = np.poly1d(coef)
    # ylims = ax.get_ylim()
    # xlims = ax.get_xlim()
    # plt.plot(xlims, poly1d_fn(xlims), line_col)
    stat_list = linregress(x,y)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    reg_y = [stat_list.slope * x_ + stat_list.intercept for x_ in xlims]
    ax.plot(xlims, reg_y, line_col)
    # xlims = ax.get_xlim()
    # ylims = ax.get_ylim()
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_aspect(1 / ax.get_data_ratio())
    # xticks, yticks = ax.get_xticks(), ax.get_yticks()
    # dict_plot = {'xticks':xticks, 'yticks':yticks, 'r_squared':stat_list.rvalue}
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.text(xlims[1] - 0.65*xlims[1], ylims[1] - 0*ylims[1],
    #         'R$^2$=' + str(round(stat_list.rvalue**2, 2)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.1*ylims[1],
    #         'R=' + str(round(stat_list.rvalue, 4)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.2*ylims[1],
    #         'p=' + str(stat_list.pvalue), fontsize=ft)
    # ax.set_aspect('equal')
    # ax.set_xlabel(xlab)
    # ax.set_ylabel(ylab)

    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # out_bn = output_dir + '/' + sn + '_corr_' + ylab + '_' + xlab
    # ip.save_png_pdf(out_bn)
    # yaml_fn = out_bn + '.yaml'
    # print(dict_plot)
    print('ylab',ylab)
    print('xlab',xlab)
    print('r squared', stat_list.rvalue**2)
    plt.show()
    plt.close()
    # with open(yaml_fn, 'w') as f:
    #     yaml.dump(dict_plot, f)

# %% codecell
# Save correlations
dims = [0.5,0.5]
size = 1
col = 'k'
line_col = 'r'
ft=6

names_spot_chan = [channel_names[i] for i in [0,1,3]]
for k in dict_keys_order:
    inds = list(eval(k))
    xlab, ylab = [names_spot_chan[i] for i in inds]
    x,y = dict_corr[k]
    fig, ax = ip.general_plot(dims=dims, ft=ft)
    ax.scatter(x,y,s=size, c=col)

    # coef = np.polyfit(x,y,1)
    # poly1d_fn = np.poly1d(coef)
    # ylims = ax.get_ylim()
    # xlims = ax.get_xlim()
    # plt.plot(xlims, poly1d_fn(xlims), line_col)
    stat_list = linregress(x,y)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    reg_y = [stat_list.slope * x_ + stat_list.intercept for x_ in xlims]
    ax.plot(xlims, reg_y, line_col)
    # xlims = ax.get_xlim()
    # ylims = ax.get_ylim()
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_aspect(1 / ax.get_data_ratio())
    # xticks, yticks = ax.get_xticks(), ax.get_yticks()
    dict_plot = {'xticks':xticks, 'yticks':yticks, 'r_squared':stat_list.rvalue}
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.text(xlims[1] - 0.65*xlims[1], ylims[1] - 0*ylims[1],
    #         'R$^2$=' + str(round(stat_list.rvalue**2, 2)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.1*ylims[1],
    #         'R=' + str(round(stat_list.rvalue, 4)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.2*ylims[1],
    #         'p=' + str(stat_list.pvalue), fontsize=ft)
    # ax.set_aspect('equal')
    # ax.set_xlabel(xlab)
    # ax.set_ylabel(ylab)
    print(ax.get_xticks(), ax.get_yticks())
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    out_bn = output_dir + '/' + sn + '_corr_' + ylab + '_' + xlab
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()




# # %% md
#
# # =============================================================================
# # ## Try with cumulative masks and psf
# # =============================================================================


# %% codecell
def generate_point_spread_function(shape=(11, 11), sigma=2.0):
    x, y = np.meshgrid(np.arange(-shape[1]//2, shape[1]//2+1), np.arange(-shape[0]//2, shape[0]//2+1))
    psf = np.exp(-(x**2 + y**2)/(2.0*sigma**2))
    psf /= np.sum(psf)
    return psf


# # %% codecell
# # Pick a psf
# shape=(50,50)
# sigma=6
# psf = generate_point_spread_function(shape=shape,sigma=sigma)
# ip.plot_image(psf, cmap='inferno')
#
# # %% codecell
# # Apply deconvolution to images
# clims = [(0,0.0025),(0,0.005), (0,0.003)]
#
# im_inches = 10
#
# spot_deconv_fmt = config['output_dir'] + '/' + config['spot_deconv_fmt']
# # for sn, thr_pk_snr in zip(sample_names, thresholds):
# for sn in sample_names:
#     # Get raw image
#     raw = ip.load_output_file(config, 'raw_fmt', sn)
#     print(sn)
#     # for s_ch in config['spot_seg']['channels']:
#     # for s_ch, thr_pk_snr in zip(config['spot_seg']['channels'], thresh_pick_snr):
#     for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
#         print(s_ch)
#         # Get proper channel
#         raw_ch = raw[:,:,s_ch]
#         # Get deconvolution
#         raw_ch_deconvolve = richardson_lucy(raw_ch, psf)
#         # Save
#         output_filename = spot_deconv_fmt.format(sample_name=sn, spot_chan=s_ch)
#         np.save(output_filename, raw_ch_deconvolve)
#         # show raw image next to thresholded image
#         im_list = [raw_ch, raw_ch_deconvolve]
#         ip.subplot_square_images(im_list, (1,2), clims=[clims_ch,clims_ch,''], im_inches=im_inches)
#         plt.show()
#         plt.close()

# %% codecell
# Apply deconvolution to images
clims = [(0,0.0025),(0,0.005), (0,0.003)]

im_inches = 10

# for sn, thr_pk_snr in zip(sample_names, thresholds):
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    print(sn)
    # for s_ch in config['spot_seg']['channels']:
    # for s_ch, thr_pk_snr in zip(config['spot_seg']['channels'], thresh_pick_snr):
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
        print(s_ch)
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        raw_ch_deconvolve = ip.load_output_file(config, 'spot_deconv_fmt', sn, spot_chan=s_ch)

        im_list = [raw_ch, raw_ch_deconvolve]
        im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
        ip.subplot_square_images(im_list, (1,2), clims=[clims_ch,()], im_inches=im_inches)
        plt.show()
        plt.close()


# # %% md
#
# # =============================================================================
# # ## Watershed segmentation
# # =============================================================================

# %% codecell
# pick sample
sn = sample_names[2]
sample_names = [sn]
print(sn)

# %% codecell
# Apply watershed
thresh_pick = [0.002,0.005,0.0025]
bgthrs = [0.001,0.004,0.001]
clims = [(0,0.0025),(0,0.005), (0,0.003)]
zc = [5000, 0]
zs = [3000, 3000]
im_inches = 10

spot_deconv_fmt = config['output_dir'] + '/' + config['spot_deconv_fmt']
# for sn, thr_pk_snr in zip(sample_names, thresholds):
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    print(sn)
    # for s_ch in config['spot_seg']['channels']:
    # for s_ch, thr_pk_snr in zip(config['spot_seg']['channels'], thresh_pick_snr):
    for s_ch, thr_pk, bgthr, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, bgthrs, clims):
        print(s_ch)
        # Get proper channel
        raw_ch = raw[:,:,s_ch][zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]]
        # watershed seeds using thresholding
        watershed_seeds = raw_ch > thr_pk
        watershed_mask = raw_ch > bgthr
        # Get segmentation
        seg_watershed = sf.watershed(-raw_ch, watershed_seeds, mask=watershed_mask)
        seg_watershed = sf.label(seg_watershed)
        # Save
        # output_filename = spot_deconv_fmt.format(sample_name=sn, spot_chan=s_ch)
        # np.save(output_filename, raw_ch_deconvolve)
        # show raw image next to thresholded image
        fig, ax, _ = ip.plot_image(raw_ch, cmap='inferno', clims=clims_ch)
        ip.plot_seg_outline(ax, seg_watershed)
        plt.show()
        plt.close()
        seg_rgb = ip.seg2rgb(seg_watershed)
        im_list = [raw_ch, seg_rgb]
        ip.subplot_square_images(im_list, (1,2), clims=[clims_ch,''], im_inches=im_inches)

# %% codecell
# Get region props from thresholded image
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% codecell
spot_watershed_fmt = config['output_dir'] + '/' + config['spot_watershed_fmt']
spot_watershed_props_fmt = config['output_dir'] + '/' + config['spot_watershed_props_fmt']
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch, thr_pk, bgthr, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, bgthrs, clims):
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        # Get mask
        watershed_seeds = raw_ch > thr_pk
        watershed_mask = raw_ch > bgthr
        # Get segmentation
        seg_watershed = sf.watershed(-raw_ch, watershed_seeds, mask=watershed_mask)
        seg_watershed = sf.label(seg_watershed)
        # Save
        output_filename = spot_watershed_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, seg_watershed)
        print('Wrote: ', output_filename)
        # get props
        props = sf.measure_regionprops(seg_watershed, raw_ch)
        output_filename = spot_watershed_props_fmt.format(sample_name=sn, spot_chan=s_ch)
        props.to_csv(output_filename)
        print('Wrote: ', output_filename)

# %% codecell
thr_pk_area = [750, 750, 500]
ylims=(0,4000)
lw=1
s=1
for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
    print(s_ch)
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    for sn in sample_names:
        print(sn)
        props = ip.load_output_file(config, 'spot_watershed_props_fmt', sn, spot_chan=s_ch)
        pa = props['area'].sort_values()
        ax.scatter(np.arange(pa.shape[0]), pa, s=s)
        xlims = ax.get_xlim()
# for tpa in thr_pk_area:
    ax.plot(xlims, [tpa]*2, color='k', lw=lw*2)
    ax.set_ylim(ylims[0],ylims[1])
    plt.show()
    plt.close()


# %% md

# Save area filtered mask

# %% codecell
spot_mask_bg_debris_fmt = config['output_dir'] + '/' + config['spot_mask_bg_debris_fmt']
spot_props_bg_debris_fmt = config['output_dir'] + '/' + config['spot_props_bg_debris_fmt']
for sn in sample_names:
    # Load raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
    # for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
        # pick proper channel
        raw_ch = raw[:,:,s_ch]
        # Load mask
        mask_old = ip.load_output_file(config, 'spot_watershed_fmt', sn, spot_chan=s_ch)
        # load regionprops
        props = ip.load_output_file(config, 'spot_watershed_props_fmt', sn, spot_chan=s_ch)
        # Mask raw image
        # Get bool for area threshold
        props['area_thresh'] = (props.area < tpa).astype(int)
        # props['area_thresh'] = (props.area < tpa).astype(int)
        # Save new regionprops
        output_filename = spot_props_bg_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
        props.to_csv(output_filename, index=False)
        # Remove filtered objects from mask
        mask_new = ip.filter_seg_objects(mask_old, props, 'area_thresh')
        print(props.shape[0], props.area_thresh.sum())
        # New mask on raw image
        # Save new mask
        output_filename = spot_mask_bg_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, mask_new)
        print(output_filename)
        # show the images side by side

# %% md

# Show filtering. From left to right: raw image, SNR thresholded mask, area thresholded mask

# %% codecell
# clims = (0,0.01)
im_inches=10

for sn in sample_names:
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
    # for s_ch in config['spot_seg']['channels']:
        raw_ch = raw[:,:,s_ch]
        mask_old = ip.load_output_file(config, 'spot_watershed_fmt', sn, spot_chan=s_ch)
        raw_mask_old = raw_ch * (mask_old > 0)
        mask_new = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
        raw_mask_new = raw_ch * (mask_new > 0)
        im_list = [raw_ch, mask_old>0, mask_new>0]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,(),()], im_inches=im_inches)

# %% codecell
# clims = (0,0.01)
zc = [4000, 3000]
zs = [3000, 3000]

for sn in sample_names:
    print(sn)
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    # for s_ch in config['spot_seg']['channels']:
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
        print(s_ch)
        raw_ch = raw[:,:,s_ch]
        mask_old = ip.load_output_file(config, 'spot_watershed_fmt', sn, spot_chan=s_ch)
        raw_mask_old = raw_ch * (mask_old > 0)
        mask_new = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
        raw_mask_new = raw_ch * (mask_new > 0)
        im_list = [raw_ch, mask_old>0, mask_new>0]
        im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
        plt.show()
        plt.close()

# %% md

# =============================================================================
# ## Second round of intensity filtering
# =============================================================================

# %% codecell
thr_pk_int = [0.002, 0.01, 0.00245]
ylims=(0,0.04)
dims=(5,5)
lw=1
s=1
for s_ch, tpi in zip(config['spot_seg']['channels'], thr_pk_int):
    print(s_ch)
    fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
    for sn in sample_names:
        print(sn)
        props = ip.load_output_file(config, 'spot_props_bg_debris_fmt', sn, spot_chan=s_ch)
        pa = props['max_intensity'].sort_values()
        ax.scatter(np.arange(pa.shape[0]), pa, s=s)
        xlims = ax.get_xlim()
# for tpa in thr_pk_area:
    ax.plot(xlims, [tpi]*2, color='k', lw=lw*2)
    ax.set_ylim(ylims[0],ylims[1])
    plt.show()
    plt.close()


# %% codecell
spot_mask_bg_int_fmt = config['output_dir'] + '/' + config['spot_mask_bg_int_fmt']
spot_props_bg_int_fmt = config['output_dir'] + '/' + config['spot_props_bg_int_fmt']

sn = sample_names[0]
raw = ip.load_output_file(config, 'raw_fmt', sn)
for s_ch, tpi in zip(config['spot_seg']['channels'], thr_pk_int):
# for s_ch, tpa in zip(config['spot_seg']['channels'], thr_pk_area):
    # pick proper channel
    raw_ch = raw[:,:,s_ch]
    # Load mask
    mask_old = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
    # load regionprops
    props = ip.load_output_file(config, 'spot_props_bg_debris_fmt', sn, spot_chan=s_ch)
    # Mask raw image
    # Get bool for area threshold
    props['int_thresh'] = (props.max_intensity > tpi).astype(int)
    # props['area_thresh'] = (props.area < tpa).astype(int)
    # Save new regionprops
    output_filename = spot_props_bg_int_fmt.format(sample_name=sn, spot_chan=s_ch)
    props.to_csv(output_filename, index=False)
    # Remove filtered objects from mask
    mask_new = ip.filter_seg_objects(mask_old, props, 'int_thresh')
    print(np.unique(mask_old).shape[0], np.unique(mask_new).shape[0])
    # New mask on raw image
    # Save new mask
    output_filename = spot_mask_bg_int_fmt.format(sample_name=sn, spot_chan=s_ch)
    np.save(output_filename, mask_new)
    print(output_filename)
    # show the images side by side

# %% codecell


print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_old = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
    raw_mask_old = raw_ch * (mask_old > 0)
    mask_new = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    im_list = [raw_ch, mask_old>0, mask_new>0]
    # im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
    plt.show()
    plt.close()


# %% codecell
# clims = (0,0.01)
zc = [3000, 0]
zs = [3000, 3000]

print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_old = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
    raw_mask_old = raw_ch * (mask_old > 0)
    mask_new = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    im_list = [raw_ch, mask_old>0, mask_new>0]
    im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,clims_ch,clims_ch], im_inches=im_inches)
    plt.show()
    plt.close()

# # %% md
#
# # =============================================================================
# # ## Colocalization
# # =============================================================================

# %% codecell
from sklearn.neighbors import NearestNeighbors

r_um = 5

r_pix = r_um / config['resolution']
neighs = {}
ch_coords = []
# Load props
for s_ch, tpas in zip(config['spot_seg']['channels'], thr_pk_area_sm):
    print(s_ch)
    props = ip.load_output_file(config, 'spot_props_bg_int_fmt', sn, spot_chan=s_ch)
    # Create kneighbors for each channel
    props.columns
    bools = (props.area_thresh.values * props.int_thresh.values) == 1
    # bools = (props.area_thresh.values * props.int_thresh.values * props.ecc_thresh.values) == 1
    props_ = props[bools]
    coords = [list(eval(c)) for c in props_.centroid.values]
    print(len(coords))
    neighs[s_ch] = NearestNeighbors(radius=r_pix).fit(coords)
    ch_coords.append(coords)

# %% codecell
# Get kneighbors in certain radius for both other channel
dict_frac_assoc = {}
for s_ch, coords in zip(config['spot_seg']['channels'], ch_coords):
    dict_frac_assoc[s_ch] = {}
    for s_ch_2 in config['spot_seg']['channels']:
        coords_both = []
        if not s_ch == s_ch_2:
            dists, inds = neighs[s_ch_2].radius_neighbors(coords)

            bool_ch = [i.shape[0]>0 for i in inds]
            count = sum(bool_ch)
            print(len(coords))
            dict_frac_assoc[s_ch][s_ch_2] = count / len(coords)


dict_frac_assoc

# count fraction assoc with each
# count frac assoc with both

# %% codecell
size=10
alpha=0.5
fig, ax = ip.general_plot(dims = (20,20))
ax.imshow(np.zeros(raw_ch.shape + (3,)))
for s_ch, tpas in zip(config['spot_seg']['channels'], thr_pk_area_sm):
    print(s_ch)
    props = ip.load_output_file(config, 'spot_props_bg_int_fmt', sn, spot_chan=s_ch)
    # Create kneighbors for each channel
    props.columns
    bools = (props.area_thresh.values * props.int_thresh.values) == 1
    props_ = props[bools]
    coords = np.array([list(eval(c)) for c in props_.centroid.values])
    print(coords.shape)
    ax.scatter(coords[:,1],coords[:,0], s=size, alpha=alpha, label=s_ch)
ax.legend()


# %% codecell
# Get kneighbors in certain radius for both other channel
# dict_frac_assoc = {}
# for s_ch, coords_ch in zip(config['spot_seg']['channels'], ch_coords):
#     dict_frac_assoc[s_ch] = {}
#     print('channel',s_ch)
#     coords_both = []
#     for s_ch_2, coords_ch_2 in zip(config['spot_seg']['channels'], ch_coords):
#         if not s_ch == s_ch_2:
#             dists, inds = neighs[s_ch_2].radius_neighbors(coords_ch)
#             bool_ch = [i.shape[0]>0 for i in inds]
#             count = sum(bool_ch)
#             frac = count / len(coords_ch)
#             dict_frac_assoc[s_ch][s_ch_2] = frac
#             print('with',s_ch_2, count, frac)
            # coords_both += coords_ch_2

    # dists, inds = NearestNeighbors(radius=r_pix).fit(coords_both).radius_neighbors(coords_ch)
    # bool_ch = [i.shape[0]>0 for i in inds]
    # count = sum(bool_ch)
    # frac = count / len(coords_ch)
    # print('both', count, frac)

# %% codecell
# barplot
ft=6
dims=(1.05, 0.4)
lw=1
line_col='k'
height=0.5
barlw=0.5
col_dict = {0:(1,1,0), 1:(0,1,1), 3:(1,0,1)}
ylims=[-1,2]
xlims=[0,1]

for s_ch, coords_ch in zip(config['spot_seg']['channels'], ch_coords):
    print(s_ch)
    fig, ax = ip.general_plot(ft=ft, dims=dims, lw=lw, col=line_col)
    data = dict_frac_assoc[s_ch]
    order = [k for k in data.keys()]
    width = [data[k] for k in order]
    y = np.arange(len(order))
    barlist = ax.barh(y, width=width, height=height, ec=line_col, lw=barlw)
    for b, k in zip(barlist, order):
        b.set_facecolor(col_dict[k])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_yticks(labels=[],ticks=[])
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    # if not s_ch == 3:
    ax.set_xticklabels([])
    out_dir = config['output_dir'] + '/spot_assoc_barplots'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_bn = out_dir + '/' + sn + '_bar_plots_spot_chan_' + str(s_ch)
    ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()


# %% md

# ==============================================================================
# ## get correlation between colors
# ==============================================================================

# %% codecell
# clims = (0,0.01)
zc = [3000, 0]
zs = [3000, 3000]

print(sn)
raw = ip.load_output_file(config, 'raw_fmt', sn)
# for s_ch in config['spot_seg']['channels']:
raw_masked = []
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    mask_new = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_mask_new = raw_ch * (mask_new > 0)
    raw_masked.append(raw_mask_new)

raw_masked = np.dstack(raw_masked)
raw_masked.shape

# %% codecell
# Divide image into regions,
shape = raw_ch.shape
region_size_um = 2
region_size = int(region_size_um / config['resolution'])
rows = np.arange(0,shape[0],region_size)
columns = np.arange(0,shape[1],region_size)
rows.shape, columns.shape


# %% codecell
# get color intensity correlation across all regions
dict_corr = {'0,1':[[],[]], '0,2':[[],[]],'1,2':[[],[]]}
dict_keys_order = ['0,1','0,2','1,2']
bool_selector = np.array([
        [1,1,0],
        [1,0,1],
        [0,1,1]
        ])
for i in tqdm(range(rows.shape[0] - 1)):
    r_0, r_1 = rows[i], rows[i+1]
    r = raw_masked[r_0:r_1,:,:]

    for j in range(columns.shape[0] - 1):
        c_0, c_1 = columns[j], columns[j+1]
        square = r[:, c_0:c_1, :]
        maxs = [np.max(square[:,:,i]) for i in range(raw_masked.shape[2])]
        bools = np.array([m > 0 for m in maxs])
        # bools = np.array([m > t for m, t in zip(maxs, thresh_chans)])

        if sum(bools) == 1:
            selec = bool_selector[bools,:].squeeze().astype(bool)
            keys_towrite = np.array(dict_keys_order)[selec]
        elif sum(bools) > 1:
            keys_towrite = dict_keys_order
        else:
            keys_towrite = []

        for key in keys_towrite:
            inds = list(eval(key))
            dict_corr[key][0].append(maxs[inds[0]])
            dict_corr[key][1].append(maxs[inds[1]])



# %% codecell
# PLot correlations
channel_names=['DNA 2','mRNA','rRNA','DNA 1']
dims = [1.,1.]
size = 1
col = 'k'
line_col = 'r'
ft=12

names_spot_chan = [channel_names[i] for i in [0,1,3]]
for k in dict_keys_order:
    inds = list(eval(k))
    xlab, ylab = [names_spot_chan[i] for i in inds]
    x,y = dict_corr[k]
    fig, ax = ip.general_plot(dims=dims, ft=ft)
    ax.scatter(x,y,s=size, c=col)

    # coef = np.polyfit(x,y,1)
    # poly1d_fn = np.poly1d(coef)
    # ylims = ax.get_ylim()
    # xlims = ax.get_xlim()
    # plt.plot(xlims, poly1d_fn(xlims), line_col)
    stat_list = linregress(x,y)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    reg_y = [stat_list.slope * x_ + stat_list.intercept for x_ in xlims]
    ax.plot(xlims, reg_y, line_col)
    # xlims = ax.get_xlim()
    # ylims = ax.get_ylim()
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_aspect(1 / ax.get_data_ratio())
    # xticks, yticks = ax.get_xticks(), ax.get_yticks()
    # dict_plot = {'xticks':xticks, 'yticks':yticks, 'r_squared':stat_list.rvalue}
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.text(xlims[1] - 0.65*xlims[1], ylims[1] - 0*ylims[1],
    #         'R$^2$=' + str(round(stat_list.rvalue**2, 2)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.1*ylims[1],
    #         'R=' + str(round(stat_list.rvalue, 4)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.2*ylims[1],
    #         'p=' + str(stat_list.pvalue), fontsize=ft)
    # ax.set_aspect('equal')
    # ax.set_xlabel(xlab)
    # ax.set_ylabel(ylab)

    # if not os.path.exists(output_dir): os.makedirs(output_dir)
    # out_bn = output_dir + '/' + sn + '_corr_' + ylab + '_' + xlab
    # ip.save_png_pdf(out_bn)
    # yaml_fn = out_bn + '.yaml'
    # print(dict_plot)
    print('ylab',ylab)
    print('xlab',xlab)
    print('r squared', stat_list.rvalue**2)
    plt.show()
    plt.close()
    # with open(yaml_fn, 'w') as f:
    #     yaml.dump(dict_plot, f)

# %% codecell
# Save correlations
dims = [0.5,0.5]
size = 1
col = 'k'
line_col = 'r'
ft=6

names_spot_chan = [channel_names[i] for i in [0,1,3]]
for k in dict_keys_order:
    inds = list(eval(k))
    xlab, ylab = [names_spot_chan[i] for i in inds]
    x,y = dict_corr[k]
    fig, ax = ip.general_plot(dims=dims, ft=ft)
    ax.scatter(x,y,s=size, c=col)

    # coef = np.polyfit(x,y,1)
    # poly1d_fn = np.poly1d(coef)
    # ylims = ax.get_ylim()
    # xlims = ax.get_xlim()
    # plt.plot(xlims, poly1d_fn(xlims), line_col)
    stat_list = linregress(x,y)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    reg_y = [stat_list.slope * x_ + stat_list.intercept for x_ in xlims]
    ax.plot(xlims, reg_y, line_col)
    # xlims = ax.get_xlim()
    # ylims = ax.get_ylim()
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_aspect(1 / ax.get_data_ratio())
    # xticks, yticks = ax.get_xticks(), ax.get_yticks()
    dict_plot = {'xticks':xticks, 'yticks':yticks, 'r_squared':stat_list.rvalue}
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.text(xlims[1] - 0.65*xlims[1], ylims[1] - 0*ylims[1],
    #         'R$^2$=' + str(round(stat_list.rvalue**2, 2)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.1*ylims[1],
    #         'R=' + str(round(stat_list.rvalue, 4)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.2*ylims[1],
    #         'p=' + str(stat_list.pvalue), fontsize=ft)
    # ax.set_aspect('equal')
    # ax.set_xlabel(xlab)
    # ax.set_ylabel(ylab)
    print(ax.get_xticks(), ax.get_yticks())
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    out_bn = output_dir + '/' + sn + '_corr_' + ylab + '_' + xlab
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()


# %% md

# ==============================================================================
# ## pixel correlation
# ==============================================================================

# %% codecell
chans_sub = [0,1]
ims = [raw[:,:,i] for i in chans_sub]
im_stack = np.dstack([im[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for im in ims])
xs, ys = [], []
for i in tqdm(range(im_stack.shape[0])):
    for j in range(im_stack.shape[1]):
        x, y = ims_zoom[i,j,:]
        xs.append(x)
        ys.append(y)

# %% codecell
dims = [5,5]
size = 1
col = 'k'
line_col = 'r'
ft=12

fig, ax = ip.general_plot(dims=dims, ft=ft)
ax.scatter(xs,ys,s=size, c=col)
stat_list = linregress(xs,ys)
xlims = ax.get_xlim()
ylims = ax.get_ylim()
reg_y = [stat_list.slope * x_ + stat_list.intercept for x_ in xlims]
ax.plot(xlims, reg_y, line_col)
ax.set_xlim(xlims[0],xlims[1])
ax.set_ylim(ylims[0],ylims[1])
ax.set_aspect(1 / ax.get_data_ratio())
print('r squared', stat_list.rvalue**2)


# %% md

# ==============================================================================
# ## venn diagram
# ==============================================================================

# %% codecell
# Divide image into regions,
region_size_um = 5

shape = raw_ch.shape
region_size = int(region_size_um / config['resolution'])
rows = np.arange(10,shape[0],region_size)
columns = np.arange(10,shape[1],region_size)
rows.shape, columns.shape


# %% codecell
# stack segmentations
raw_masked = []
for s_ch in config['spot_seg']['channels']:
    props = ip.load_output_file(config, 'spot_props_bg_int_fmt', sn, spot_chan=s_ch)
    bools = (props.area_thresh.values * props.int_thresh.values) == 1
    props_filt = props[bools]
    chan = np.zeros(shape)
    centroids = [[int(l) for l in eval(p)] for p in props_filt['centroid'].values]
    print(len(centroids))
    x,y = np.array(centroids).T
    chan[x,y] = 1
    # mask_new = ip.load_output_file(config, 'spot_mask_bg_int_fmt', sn, spot_chan=s_ch)
    raw_masked.append(chan)

raw_masked = np.dstack(raw_masked)
raw_masked.shape


# %% codecell
# get color intensity correlation across all regions
dict_counts = {f'{i:03b}':0 for i in range(1,8)}
for i in tqdm(range(rows.shape[0] - 1)):
    r_0, r_1 = rows[i], rows[i+1]
    r = raw_masked[r_0:r_1,:,:]
    for j in range(columns.shape[0] - 1):
        c_0, c_1 = columns[j], columns[j+1]
        square = r[:, c_0:c_1, :]
        bools = [square[:,:,k].any()*1 for k in range(square.shape[2])]
        if any(bools):
            group = ''.join([str(b) for b in bools]).zfill(3)
            dict_counts[group] += 1

dict_counts

# %% codecell
# Plot bar graph of values

list('101')
widths = np.zeros(3)
for k, v in dict_counts.items():
    group = sum([int(i) for i in list(k)])
    widths[group-1] += v
widths

# %% codecell
ft=6
dims=(1.05, 0.4)
lw=1
color='k'
line_col='k'
height=0.5
barlw=0.5
col_dict = {0:(1,1,0), 1:(0,1,1), 3:(1,0,1)}
ylims=[-1,2]
xlims=[0,1]

fig, ax = ip.general_plot(ft=ft, dims=dims, lw=lw, col=line_col)
y = np.arange(len(widths))
barlist = ax.barh(y, width=widths, height=height, fc=color, ec=line_col, lw=barlw)
# ax.set_ylim(ylims[0],ylims[1])
# ax.set_xlim(xlims[0], xlims[1])
ax.set_yticks(labels=[],ticks=[])
ax.spines['bottom'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# if not s_ch == 3:
ax.set_xticklabels([])
out_dir = config['output_dir'] + '/spot_assoc_barplots'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_bn = out_dir + '/' + sn + '_bar_plots_general'
ip.save_png_pdf(out_bn)
plt.show()
plt.close()




# %% md

# ==============================================================================
# ## Plot spot density on map
# ==============================================================================

# %% codecell
# Get cell
c_ch = config['cell_seg']['channels'][0]
raw = ip.load_output_file(config, 'raw_fmt', sn)
cell = raw[:,:,c_ch]
cell_pix_values = np.sort(np.ravel(cell))
# %% codecell
# PLot intensities
thresh_cell=0.005
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
r_um = 20
r_pix = int(r_um / config['resolution'])




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


# # %% codecell
# manually draw mask in fiji
edge_mask_fn = config['output_dir'] + '/edge_mask/' + sn + 'edgemask.tif'

# Get edge of mask
cell_mask_bound = ip.find_boundaries(cell_mask, mode='inner')
# Convolve with circle
cell_mask_edge = fftconvolve(cell_mask_bound, circle_mask)
# multiply by mask
cell_mask_edge *= np.pad(cell_mask, np.int(r_pix), mode='edge')
cell_mask_edge =

# plot
ip.plot_image(cell_mask_edge > 0)


# %% codecell
# For each pixel coordinate
thresh_area_frac = 0.9

thresh_circle_area = area_circle_pix * thresh_area_frac
cell_mask_pad = np.pad(cell_mask, np.int(r_pix), mode='edge')
area_list = []
um_sq_per_pix = config['resolution']**2
for i,j in tqdm(pix_coords):
    # Get bbox
    i += r_pix
    j += r_pix
    bbox = cell_mask_pad[i-r_pix: i+r_pix+1, j-r_pix: j+r_pix+1]
    if np.sum(bbox) < thresh_circle_area:
        # multiply bbox by circle
        reduced_mask = circle_mask * bbox
        area = np.sum(reduced_mask) * um_sq_per_pix
    else:
        area = area_circle
    # add to area list
    area_list.append(area)

# %% codecell
# plot edge correction
cell_edge_map = np.zeros(cell.shape)
cell_edge_map[pix_inds] = area_list
fig, ax, cbar = ip.plot_image(cell_edge_map)
background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
background = np.dstack(background)
ax.imshow(background)

# %% codecell
# Calculate spot densities at subset of pixels
subset=2
chan_densities = []
spot_coords_sub = np.array(spot_coords)[::subset]
area_list_sub = np.array(area_list)[::subset]
for s_ch in config['spot_seg']['channels']:
    print(s_ch)
    # Get spot coords
    props = ip.load_output_file(config, 'spot_props_bg_int_fmt', sn, spot_chan=s_ch)
    bools = (props.area_thresh.values * props.int_thresh.values) == 1
    props_filt = props[bools]
    spot_coords = [list(eval(c)) for c in props_filt.centroid.values]
    print(len(spot_coords))
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
    pix_density = np.array(pix_counts) / np.array(area_list)
    chan_densities.append(pix_density)


# %% codecell
# Plot
clims = (0, np.max(chan_densities))
for s_ch, pix_density in zip(config['spot_seg']['channels'][:1], chan_densities):
    print(s_ch)
    cell_spot_density = np.zeros(cell.shape)
    cell_spot_density[pix_inds] = pix_density
    fig, ax, cbar = ip.plot_image(
            cell_spot_density,
            cmap='cividis',
            scalebar_resolution=config['resolution'],
            clims=clims,
            cbar_ori='vertical'
            )
    background = [np.zeros(cell.shape)]*3 + [~cell_mask*1]
    background = np.dstack(background)
    ax.imshow(background)
    # plt.figure(fig)
    # out_dir = config['output_dir'] + '/density_maps'
    # if not os.path.exists(out_dir): os.makedirs(out_dir)
    # output_bn = out_dir + '/' + sn + '_spot_density_map_chan_' + str(s_ch)
    # ip.save_png_pdf(output_bn)
    # plt.figure(cbar)
    out_dir = config['output_dir'] + '/density_maps'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    output_bn = out_dir + '/' + sn + '_spot_density_map_chan_' + str(s_ch) + '_cbar'
    ip.save_png_pdf(output_bn)
    plt.show()
    plt.close()

# %% md

# ==============================================================================
# ## Violin of spot intensity
# ==============================================================================

# %% codecell



# %% codecell
