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
# from tqdm import tqdm
gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2023_04_06_mefe/MeTAFISH'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

# Go into your configuration file and adjust all of the 'Paths' so they are relative to the workdir.

# Also adjust the 'Inputs' parameters to fit the images.

# %% codecell
config_fn = 'config_meta.yaml' # relative path to config file from workdir

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

in_glob = config['input_dir'] + '/' + config['input_regex'] + config['input_ext']

input_filenames = glob.glob(in_glob)
input_filenames.sort()
print(in_glob)
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

# %% codecell
# pick a sample
sample_names = [sample_names[0]]

clims = [(0.001,0.01)]

im_inches = 20

for sn in sample_names:
    print(sn)
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        print(s_ch)
        ip.plot_image(raw_ch, cmap='inferno',im_inches=im_inches, clims=clims_ch)
        plt.show()
        plt.close()

# %% md
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
cmap = ip.get_cmap('inferno')(np.linspace(0,1,256))
cmap[0] = [0.5,0.5,0.5,1]
cmap = ip.ListedColormap(cmap)
dims=(2,2)
lw=1
ft=6
im_inches=20

# bg_df = ip.load_output_file(config, 'spot_vals_bgrough_fmt')
fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
ticks = []
for k, sn in enumerate(sample_names):
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    shift = k
    ticks.append(shift)
    for l, (s_ch, clims_ch) in enumerate(zip(config['spot_seg']['channels'], clims)):
        mean, std = bg_df[sn + '_spotchan_' + str(s_ch)].values
        shift2 = shift + l*0.2
        ax.errorbar(shift2, mean, yerr=std, marker='.')
        print(sn)
        raw_ch = raw[:,:,s_ch]
        bg_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=s_ch)
        raw_mask = raw_ch * bg_mask
        im_list = [raw_ch, raw_mask]
        fig2, ax2, cbar = ip.subplot_square_images(im_list, (1,2), clims=[clims_ch,clims_ch], cmaps=[cmap, cmap], im_inches=im_inches)
        plt.figure(fig2)
plt.figure(fig)
ax.set_xticks(ticks)
ax.set_xticklabels([])
plt.show()

# # %% md
#
# # Get threshold curves
#
# # %% codecell
# blims = (0,20)
# n_bthr = 20
# bthreshs = np.linspace(blims[0], blims[1], n_bthr)
#
# # spot_vals_bg_curve_fmt = config['output_dir'] + '/' + config['spot_vals_bg_curve_fmt']
# b_curves_df = pd.DataFrame([])
# for sn in tqdm(sample_names):
#     # Get raw image
#     raw = ip.load_output_file(config, 'raw_fmt', sn)
#     # Get proper channel
#     for s_ch in config['spot_seg']['channels']:
#         raw_ch = raw[:,:,s_ch]
#         # Get background mask
#         # bg_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=s_ch)
#         # Get background valeue
#         bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
#         # Get SNR image
#         raw_snr = raw_ch / bg_val
#         # Get the number of pixels before thresholding
#         n = raw_snr.ravel().shape[0]
#         # Calculate ratio and add to curve values
#         curve = [raw_snr[raw_snr > thr].shape[0] / n for thr in bthreshs]
#         b_curves_df[sn + '_spotchan_' + str(s_ch)] = curve
#         # Save threshold curve values
# # output_filename = spot_vals_bg_curve_fmt
# # curves_df.to_csv(output_filename, index=False)
# # curves_df.shape
#
# # %% md
#
# # Plot curves
#
# # %% codecell
# # Pick a threshold
# # thr_pk_snr = [40]
# thresh_pick_snr = [3,5,3]
# xlims=(1,20)
# ylims=[0,0.25]
# dims=(4,4)
# colors = ip.get_cmap('tab10').colors
# cols = colors[:3]
#
#
# # spot_curve_bg_fmt = config['output_dir'] + '/{sample_name}_chan_{spot_chan}_curve_bg'
# # curves_df = ip.load_output_file(config, 'spot_vals_bg_curve_fmt')
# # Start a figure
# for s_ch, thr_pk_snr, clr in zip(config['spot_seg']['channels'], thresh_pick_snr, cols):
#     fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
#     for sn in sample_names:
#         # for sn, thr_pk_snr in zip(sample_names,thresh_pick_snr):
#     # print(sn)
#         # Get curve
#         curve = b_curves_df[sn + '_spotchan_' + str(s_ch)].values
#         # Plot curve and threshold
#         ax.plot(bthreshs, curve, lw=lw, color=clr)
#         ax.plot([thr_pk_snr]*2, [0,1], color=clr, lw=lw)
#         ax.plot(list(xlims), [0,0],'k',lw=lw*0.5)
#         # adjust plot
#     ax.set_xlim(xlims)
#     ax.set_ylim(ylims)
#     # # save plot
#     # output_basename = spot_curve_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
#     # ip.save_png_pdf(output_basename)
#     # show plot
#     plt.show()

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
# sample_names = [sample_names[2]]


# %% md

# Plot ordered pixel intensities

# %% codecell
bg_thr = 0.001
thresh_pick = [0.004]


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
        bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
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
zc = [1000, 2000]
zs = [1000, 1000]
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
        bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
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

# %% codecell
# Get mask in napari and write to thes filenames
# from skimage.io import imread
manual_mask_shapes_fmt = config['output_dir'] + '/' + config['spot_manual_mask_shapes_fmt']
for sn in sample_names:
    for s_ch in config['spot_seg']['channels']:
        print(os.getcwd() + '/' + manual_mask_shapes_fmt.format(sample_name=sn, spot_chan=s_ch))

# %% codecell
# show the mask
im_inches=20
for sn in sample_names:
    for s_ch in config['spot_seg']['channels']:
        mask = ip.load_output_file(config, 'spot_manual_mask_shapes_fmt', sn, spot_chan=s_ch)
        ip.plot_image(mask, im_inches=im_inches)

# %% codecell
# mask the image
manual_mask_fmt = config['output_dir'] + '/' + config['spot_manual_mask_fmt']
for sn in sample_names:
    for s_ch in config['spot_seg']['channels']:
        mask = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=s_ch)
        shapes = ip.load_output_file(config, 'spot_manual_mask_shapes_fmt', sn, spot_chan=s_ch)
        mask_new = mask * (shapes == 0)
        np.save(manual_mask_fmt.format(sample_name=sn, spot_chan=s_ch), mask_new)

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
        mask_new = ip.load_output_file(config, 'spot_manual_mask_fmt', sn, spot_chan=s_ch)
        raw_mask_new = raw_ch * (mask_new > 0)
        im_list = [raw_ch, mask_old>0, mask_new>0]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,(),()], im_inches=im_inches)


# %% md

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
        mask = ip.load_output_file(config, 'spot_manual_mask_fmt', sn, spot_chan=s_ch)
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
thr_pk_area = [600]
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
        mask_old = ip.load_output_file(config, 'spot_manual_mask_fmt', sn, spot_chan=s_ch)
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
        mask_old = ip.load_output_file(config, 'spot_manual_mask_fmt', sn, spot_chan=s_ch)
        raw_mask_old = raw_ch * (mask_old > 0)
        mask_new = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
        raw_mask_new = raw_ch * (mask_new > 0)
        im_list = [raw_ch, mask_old>0, mask_new>0]
        ip.subplot_square_images(im_list, (1,3), clims=[clims_ch,(),()], im_inches=im_inches)

# %% codecell
# clims = (0,0.01)
zc = [1000, 2000]
zs = [1000, 1000]

for sn in sample_names:
    print(sn)
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    # for s_ch in config['spot_seg']['channels']:
    for s_ch, thr_pk, clims_ch in zip(config['spot_seg']['channels'], thresh_pick, clims):
        print(s_ch)
        raw_ch = raw[:,:,s_ch]
        mask_old = ip.load_output_file(config, 'spot_manual_mask_fmt', sn, spot_chan=s_ch)
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
thr_pk_int = [0.003]

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
    print(props.shape[0], props.int_thresh.sum())
    # New mask on raw image
    # Save new mask
    output_filename = spot_mask_bg_int_fmt.format(sample_name=sn, spot_chan=s_ch)
    np.save(output_filename, mask_new)
    print(output_filename)
    # show the images side by side

# %% codecell
# clims = (0,0.01)
zc = [1000, 2000]
zs = [1000, 1000]

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
thr_pk_ecc = [0.85]
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
# clims = (0,0.01)
zc = [1000, 2000]
zs = [1000, 1000]

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

# %% codecell
# clims = (0,0.01)

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
# # ## Final check
# # =============================================================================

# %% codecell
clims = [(0.001,0.01)]

marker='o'
size=100
im_inches=20
cmap='inferno'
out_fmt = os.path.splitext(config['output_dir'] + '/' + config['plot_final_spots'])[0]

for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    # im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    fig, ax, _ = ip.plot_image(raw_ch, clims=clims_ch, im_inches=im_inches, cmap=cmap)
    props = ip.load_output_file(config, 'spot_props_bg_ecc_fmt', sn, spot_chan=s_ch)
    coords = np.array([list(eval(c)) for c in props.centroid.values])
    ax.scatter(coords[:,1],coords[:,0], marker=marker,facecolor=(0,0,0,0), s=size, edgecolor='g')
    output_basename = out_fmt.format(sample_name=sn, spot_chan=s_ch)
    out_dir = os.path.split(output_basename)[0]
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    plt.sca(ax)
    ip.save_png_pdf(output_basename)
    print(output_basename)
    plt.show()
    plt.close()



# %% md

# =============================================================================
# ## deconvolve the point spread funciton
# =============================================================================


# %% codecell
import numpy as np
from scipy.signal import convolve2d, fftconvolve

def pad_kernel(input_array, filter_kernel):
    # Determine the required padding for the filter kernel
    pad_rows = (input_array.shape[0] - filter_kernel.shape[0] + 1) // 2
    pad_cols = (input_array.shape[1] - filter_kernel.shape[1] + 1) // 2
    return np.pad(filter_kernel, ((pad_rows, pad_rows-1), (pad_cols, pad_cols-1)), mode='constant')


def deconvolve_filter(input_array, filter_kernel):
    # Normalize the filter kernel
    filter_kernel = filter_kernel / np.sum(filter_kernel)
    input_array = input_array / np.sum(input_array)

    # filter_kernel_padded = pad_kernel(input_array, filter_kernel)

    # Determine the required padding for the input array
    pad_rows = filter_kernel.shape[0] - 1
    pad_cols = filter_kernel.shape[1] - 1
    input_array_padded = np.pad(input_array, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode='constant')

    # Perform FFT convolution to obtain the Fourier transform of the input and filter kernel
    print('FFT...')
    input_array_fft = np.fft.fftshift(np.fft.fft2(input_array_padded))
    filter_kernel_fft = np.fft.fftshift(np.fft.fft2(filter_kernel, s=input_array_padded.shape))

    # Divide the Fourier transform of the input by the Fourier transform of the filter kernel
    print('Divide...')
    output_array_fft = np.divide(input_array_fft, filter_kernel_fft)

    # Perform inverse FFT and unshift to obtain the deconvolved output array
    print('iFFT...')
    output_array = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(output_array_fft))))

    return output_array


def generate_point_spread_function(shape=(11, 11), sigma=2.0):
    x, y = np.meshgrid(np.arange(-shape[1]//2, shape[1]//2+1), np.arange(-shape[0]//2, shape[0]//2+1))
    psf = np.exp(-(x**2 + y**2)/(2.0*sigma**2))
    psf /= np.sum(psf)
    return psf



# %% codecell
# Pick a psf
shape=(50,50)
sigma=6
psf = generate_point_spread_function(shape=shape,sigma=sigma)

r = 30
dims=(10,5)
alpha=0.5
lw=0.5
linecol='k'
psfcol='r'
raw = ip.load_output_file(config, 'raw_fmt', sn)
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    fig, ax = ip.general_plot(dims=dims)
    raw_ch = raw[:,:,s_ch]
    # im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    props = ip.load_output_file(config, 'spot_props_bg_ecc_fmt', sn, spot_chan=s_ch)
    i = 0
    for c in props.centroid.values:
        c = [int(c_) for c_ in eval(c)]
        y = raw_ch[c[0]-r:c[0]+r, c[1]]
        y_norm = y / np.max(y)
        x = np.arange(y.shape[0])
        ax.plot(x,y_norm, color=linecol, alpha=alpha, lw=lw)
psf_2d = psf[0:shape[0], shape[1]//2]
psf_2d_norm = psf_2d / np.max(psf_2d)
pad_psf = (x.shape[0] - shape[0] + 1) // 2
psf_pad = np.pad(psf_2d_norm, [(pad_psf, pad_psf)], mode='constant')
ax.plot(x, psf_pad, color=psfcol, lw=lw)


# %% codecell
# get one spot as an image
r=30
raw = ip.load_output_file(config, 'raw_fmt', sn)
single_spots = []
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    print(s_ch)
    raw_ch = raw[:,:,s_ch]
    # im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    props = ip.load_output_file(config, 'spot_props_bg_ecc_fmt', sn, spot_chan=s_ch)
    ints = props.max_intensity.values.sort()
    c = props.loc[props.max_intensity == props.max_intensity.max(), 'centroid'].values[0]
    c = [int(c_) for c_ in eval(c)]
    sp = raw_ch[c[0]-r:c[0]+r,c[1]-r:c[1]+r]
    single_spots.append(sp)
    ip.plot_image(sp, cmap='inferno')




# %% codecell
# deconvolve the image
raw_deconvolve = []
for s_ch, clims_ch, sp in zip(config['spot_seg']['channels'], clims, single_spots):
    # raw_ch = raw[:,:,s_ch]
    sp_dec = deconvolve_filter(sp, psf)
    raw_deconvolve.append(sp_dec)
    ip.subplot_square_images([sp, pad_kernel(sp, psf), sp_dec], (1,3), clims=((), (), ()))


# %% codecell
# richardson lucy deconvolution
from skimage.restoration import richardson_lucy
raw_deconvolve = []
for s_ch, clims_ch, sp in zip(config['spot_seg']['channels'], clims, single_spots):
    # raw_ch = raw[:,:,s_ch]
    sp_dec = richardson_lucy(sp, psf)
    raw_deconvolve.append(sp_dec)
    ip.subplot_square_images([sp, pad_kernel(sp, psf), sp_dec], (1,3), clims=((), (), ()))


# %% codecell
# full image
raw_deconvolve = []
for s_ch, clims_ch in zip(config['spot_seg']['channels'], clims):
    raw_ch = raw[:,:,s_ch]
    raw_deconvolve.append(richardson_lucy(raw_ch, psf))


# %% codecell
im_inches = 20
for s_ch, clims_ch, raw_ch_deconvolve in zip(config['spot_seg']['channels'], clims, raw_deconvolve):
    raw_ch = raw[:,:,s_ch]
    ip.subplot_square_images([raw_ch, raw_ch_deconvolve], (1,2), clims=(clims_ch, clims_ch), im_inches=im_inches)

# %% codecell
zc = [1000, 2000]
zs = [1000, 1000]

im_inches=20

clims_dec = [(0,0.01)]
for s_ch, clims_ch, clims_ch_dec, raw_ch_deconvolve in zip(config['spot_seg']['channels'], clims, clims_dec, raw_deconvolve):
    # raw_ch = raw[:,:,s_ch]
    rcd = raw_ch_deconvolve
    # rcd = np.flip(raw_ch_deconvolve, axis=(0))
    im_list = [raw_ch, rcd]
    im_list = [i[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]] for i in im_list]
    ip.subplot_square_images(im_list, (1,2), clims=(clims_ch, clims_ch_dec), im_inches=im_inches)
