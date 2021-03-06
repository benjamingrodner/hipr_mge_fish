# %% md

# # Figure 3c: Gel embedding comparison. Generate background mask.

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
gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3b'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

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
n_cores = 2  # number of allowed cores for the snakemake to use
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
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3c
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
        bg_mask = sf.get_background_mask(
                    raw_ch, bg_smoothing=bgs, n_clust_bg=nc, top_n_clust_bg=tnc
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
clims = (0,0.005)
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
blims = (0,50)
n_bthr = 100
bthreshs = np.linspace(blims[0], blims[1], n_bthr)

# spot_vals_bg_curve_fmt = config['output_dir'] + '/' + config['spot_vals_bg_curve_fmt']
curves_df = pd.DataFrame([])
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    # Get proper channel
    raw_ch = raw[:,:,s_ch]
    for s_ch in config['spot_seg']['channels']:
        # Get background mask
        bg_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=s_ch)
        # Get background valeue
        bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
        # Get SNR image
        raw_snr = raw_ch / bg_val
        # Get the number of pixels before thresholding
        n = raw_snr[bg_mask].shape[0]
        # Calculate ratio and add to curve values
        curve = [raw_snr[raw_snr > thr].shape[0] / n for thr in bthreshs]
        curves_df[sn + '_spotchan_' + str(s_ch)] = curve
        # Save threshold curve values
# output_filename = spot_vals_bg_curve_fmt
# curves_df.to_csv(output_filename, index=False)

# %% md

# Plot curves

# %% codecell
# Pick a threshold
thresh_pick_snr = [5,25]
xlims=(0,50)
ylims=[0,1]
dims=(4,4)

# spot_curve_bg_fmt = config['output_dir'] + '/{sample_name}_chan_{spot_chan}_curve_bg'
# curves_df = ip.load_output_file(config, 'spot_vals_bg_curve_fmt')
# Start a figure
fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
for sn, thr_pk_snr in zip(sample_names,thresh_pick_snr):
    print(sn)
    for s_ch in config['spot_seg']['channels']:
        # Get curve
        curve = curves_df[sn + '_spotchan_' + str(s_ch)].values
        # Plot curve and threshold
        ax.plot(bthreshs, curve, lw=lw)
        ax.plot([thr_pk_snr]*2, [0,1], 'k', lw=lw)
        ax.plot(list(xlims), [0,0],'k',lw=lw*0.5)
        # adjust plot
ax.set_xlim(xlims)
ax.set_ylim(ylims)
# # save plot
# output_basename = spot_curve_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
# ip.save_png_pdf(output_basename)
# show plot
plt.show()

# %% md

# Show pre-thresholded images vs post

# %% codecell
# Pick clims
clims = (0,5)

spot_mask_bg_fmt = config['output_dir'] + '/' + config['spot_mask_bg_fmt']
for sn, thr_pk_snr in zip(sample_names,thresh_pick_snr):
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch in config['spot_seg']['channels']:
        # Get proper channel
        raw_ch = raw[:,:,s_ch]
        # Get background valeue
        bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
        # Get SNR image
        raw_snr = raw_ch / bg_val
        # Get mask
        mask = raw_snr > thr_pk_snr
        # Label regions in mask
        mask_l = sf.label(mask)
        # Save mask
        output_filename = spot_mask_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, mask_l)
        # Mask image
        raw_mask = raw_snr * mask
        # show raw image next to thresholded image
        im_list = [raw_snr, raw_mask, mask]
        ip.subplot_square_images(im_list, (1,3), clims=[clims,clims,''])

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
        mask = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=ch)
        # Get regionprops in mask
        props = sf.measure_regionprops(mask, raw_ch)
        # Save regionprops
        output_filename = spot_props_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
        props.to_csv(output_filename, index=False)
        print(output_filename)


# %% md

# Get area threshold curves

# %% codecell
alims = (0,5000)
n_athr = 5000
athreshs = np.linspace(alims[0], alims[1], n_athr)
# spot_vals_debris_curve_fmt: spot_mask/fig_3c_spot_vals_bgrough.csv
# spot_vals_debris_curve_fmt = config['output_dir'] + '/' + config['spot_vals_debris_curve_fmt']
curves_df = pd.DataFrame([])
for sn in sample_names:
    for s_ch in config['spot_seg']['channels']:
        # Get regionprops
        props = ip.load_output_file(config, 'spot_props_bg_fmt', sn, spot_chan=ch)
        # Get curve value at threshold
        n = props.shape[0]
        curve = [props[props['area'] < thr].shape[0] / n for thr in athreshs]
        curves_df[sn + '_spotchan_' + str(s_ch)] = curve
# Save curves
# output_filename = spot_vals_debris_curve_fmt
# curves_df.to_csv(output_filename, index=False)

# %% md

# Plot curves

# %% codecell
thr_pk_area = 1500  # Maximum object size (in pixels)

xlims=(0,5000)
ylims=[0.95,1.01]
dims=(4,4)
h=1
# spot_curve_debris_fmt = config['output_dir'] + '/fig_3c_curve_debris'
# spot_vals_debris_curve_fmt = config['output_dir'] + '/' + config['spot_vals_debris_curve_fmt']
# curves_df = ip.load_output_file(config, 'spot_vals_debris_curve_fmt')
# Start a figure
fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
for sn in sample_names:
    for s_ch in config['spot_seg']['channels']:
        # Get curve
        curve = curves_df[sn + '_spotchan_' + str(s_ch)].values
        # Plot curve and threshold
        ax.plot(athreshs, curve, lw=lw)
ax.plot([thr_pk_area]*2, [0,h], 'k', lw=lw)
ax.plot(list(xlims), [h,h],'k',lw=lw*0.5)
# adjust plot
ax.set_xlim(xlims)
ax.set_ylim(ylims)
# save plot
# output_basename = spot_curve_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
# ip.save_png_pdf(output_basename)
# show plot
plt.show()

# %% md

# Save area filtered mask

# %% codecell
spot_mask_bg_debris_fmt = config['output_dir'] + '/' + config['spot_mask_bg_debris_fmt']
spot_props_bg_debris_fmt = config['output_dir'] + '/' + config['spot_props_bg_debris_fmt']
for sn in sample_names:
    # Load raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch in config['spot_seg']['channels']:
        # pick proper channel
        raw_ch = raw[:,:,s_ch]
        # Load mask
        mask_old = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=ch)
        # load regionprops
        props = ip.load_output_file(config, 'spot_props_bg_fmt', sn, spot_chan=ch)
        # Mask raw image
        # Get bool for area threshold
        props['area_thresh'] = (props.area < thr_pk_area).astype(int)
        # Save new regionprops
        output_filename = spot_props_bg_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
        props.to_csv(output_filename, index=False)
        # Remove filtered objects from mask
        mask_new = ip.filter_seg_objects(mask_old, props, 'area_thresh')
        # New mask on raw image
        # Save new mask
        output_filename = spot_mask_bg_debris_fmt.format(sample_name=sn, spot_chan=s_ch)
        np.save(output_filename, mask_new)
        # show the images side by side

# %% md

# Show filtering. From left to right: raw image, SNR thresholded mask, area thresholded mask

# %% codecell
clims = (0,0.01)
for sn in sample_names:
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for s_ch in config['spot_seg']['channels']:
        raw_ch = raw[:,:,s_ch]
        mask_old = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=ch)
        raw_mask_old = raw_ch * (mask_old > 0)
        mask_new = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=ch)
        raw_mask_new = raw_ch * (mask_new > 0)
        im_list = [raw_ch, mask_old, mask_new]
        ip.subplot_square_images(im_list, (1,3), clims=[clims,clims,clims])

# %% md

# =============================================================================
# ## Plots
# =============================================================================

# Get number of 16s pixels

# %% codecell
bgs=5
nc=5
tnc=4

spot_mask_bg_rough_fmt = config['output_dir'] + '/' + config['spot_mask_bg_rough_fmt']
# spot_vals_bgrough_fmt = config['output_dir'] + '/' + config['spot_vals_bgrough_fmt']
cpixcount_df = pd.DataFrame([])
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    for c_ch in config['cell_seg']['channels']:
        # Get proper channel
        raw_ch = raw[:,:,c_ch]
        print(raw_ch.shape)
        # Get rough background mask
        cp_mask = sf.get_background_mask(
                    raw_ch, bg_smoothing=bgs, n_clust_bg=nc, top_n_clust_bg=tnc
                    )
        # Save rough background mask
        output_filename = spot_mask_bg_rough_fmt.format(sample_name=sn, spot_chan=c_ch)
        np.save(output_filename, cp_mask)
        print(cp_mask.shape)
        print(output_filename)
        # Get the mean value of background
        cp = raw_ch[cp_mask].shape[0]
        cpixcount_df[sn + '_spotchan_' + str(c_ch)] = [cp]
# Save background values
# output_filename = spot_vals_bgrough_fmt.format(sample_name=sn, spot_chan=s_ch)
# bg_df.to_csv(output_filename)
# print(output_filename)


# %% codecell
# bg_df = ip.load_output_file(config, 'spot_vals_bgrough_fmt')
clims = (0, 0.05)
fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
ticks = []
for k, sn in enumerate(sample_names):
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    shift = k
    ticks.append(shift)
    for l, c_ch in enumerate(config['cell_seg']['channels']):
        mean = cpixcount_df[sn + '_spotchan_' + str(c_ch)].values
        shift2 = shift + l*0.2
        ax.bar(shift2, mean)
        print(sn)
        raw_ch = raw[:,:,c_ch]
        raw_ch.shape
        bg_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=c_ch)
        raw_mask = raw_ch * bg_mask
        bg_mask.shape
        im_list = [raw_ch, raw_mask]
        fig2, ax2, cbar = ip.subplot_square_images(im_list, (1,2), clims=[clims,clims], cmaps=[cmap, cmap])
        plt.figure(fig2)
plt.figure(fig)
ax.set_xticks(ticks)
ax.set_xticklabels([])
plt.show()


# %% md

# Plot number of pixels that are debris normalized by number of 16s pixels.

# %% codecell
# pixels removed
xlims=(0,10000)
ylims=[0.95,1.05]
lw=1
ft=6
dims=(0.75,0.75)
h=1
colors = ip.get_cmap('tab10').colors
cols = [colors[0], colors[0]]
linestyles = ('solid','dotted')
sample_names_reversed = list(reversed(sample_names))
debris_bar_fmt = config['output_dir'] + '/3c_debris_barplot'
fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
for i, (sn, c, ls) in enumerate(zip(sample_names_reversed, cols, linestyles)):
    print(sn)
    for s_ch in config['spot_seg']['channels']:
        bg_mask = ip.load_output_file(config, 'spot_mask_bg_fmt', sn, spot_chan=s_ch)
        c_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=c_ch)
        pix_mask = np.sum(bg_mask>0)
        bg_mask_area = ip.load_output_file(config, 'spot_mask_bg_debris_fmt', sn, spot_chan=s_ch)
        pix_mask_area = np.sum(bg_mask_area > 0)
        pix_16s = np.sum(c_mask > 0)
        pix_removed = (pix_mask - pix_mask_area) / pix_16s
        ax.bar(i, pix_removed, ls=ls, fill=False, ec=c)
ax.set_xticklabels([])
# save plot
output_basename = debris_bar_fmt.format(sample_name=sn, spot_chan=s_ch)
ip.save_png_pdf(output_basename)
print(output_basename)



# %% md

# Plot fraction of spot pixels vs snr threshold value

# Get threshold curves

# %% codecell
slims = (0,150)
n_sthr = 300
sthreshs = np.linspace(slims[0], slims[1], n_sthr)

curves_df = pd.DataFrame([])
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    # Get proper channel
    raw_ch = raw[:,:,s_ch]
    for s_ch in config['spot_seg']['channels']:
        # Get background mask
        props = ip.load_output_file(config, 'spot_props_bg_debris_fmt', sn, spot_chan=s_ch)
        bg_val = bg_df[sn + '_spotchan_' + str(s_ch)].values[0]
        props['snr'] = props.max_intensity / bg_val
        # Get the number of pixels before thresholding
        n = props.shape[0]
        # Calculate ratio and add to curve values
        curve = [props[props.snr > thr].shape[0] / n for thr in sthreshs]
        curves_df[sn + '_spotchan_' + str(s_ch)] = curve
        # Save threshold curve values

# %% codecell
# Pick a threshold
xlims=(0,100)
ylims=[0,1.1]
dims=(4,4)
lw=1
ft=6
dims=(1,0.5)

colors = ip.get_cmap('tab10').colors
cols = [colors[0], colors[0]]
linestyles = ('dotted','solid')
spot_curve_bg_fmt = config['output_dir'] + '/{sample_name}_chan_{spot_chan}_curve_spotsnr'
# curves_df = ip.load_output_file(config, 'spot_vals_bg_curve_fmt')
# Start a figure
fig, ax = ip.general_plot(dims=dims, lw=lw, ft=ft)
for sn, c, ls in zip(sample_names, cols, linestyles):
    print(sn)
    for s_ch in config['spot_seg']['channels']:
        # Get curve
        curve = curves_df[sn + '_spotchan_' + str(s_ch)].values
        # Plot curve and threshold
        ax.plot(sthreshs, curve, lw=lw, color=c, ls=ls)
        # ax.plot([thr_pk_snr]*2, [0,1], 'k', lw=lw)
        # ax.plot(list(xlims), [0,0],'k',lw=lw*0.5)
        # adjust plot
ax.set_xlim(xlims)
# ax.set_ylim(ylims)
# save plot
output_basename = spot_curve_bg_fmt.format(sample_name=sn, spot_chan=s_ch)
ip.save_png_pdf(output_basename)
print(output_basename)
# show plot
plt.show()
