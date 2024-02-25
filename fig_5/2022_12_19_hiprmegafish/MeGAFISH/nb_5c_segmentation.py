# %% md

# # Figure 5c: Image Segmentation for termL2 HiPR-MGE-FISH

# Used "hiprfish_imaging_py38" conda environment

# Run after nb_5c_backgroundmask.py

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
gc.enable()  # Garbage cleanup

# %% md

# Move to the working directory (workdir) you want.

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2022_12_19_hiprmegafish/MeGAFISH'

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
%load_ext autoreload
%autoreload 2

sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip
import segmentation_func as sf
import spot_funcs as spf

# %% md

# Get sample names

# %% codecell
sample_names = pd.read_csv(config['input_table_fn']).sample_name.values
sample_names

# %% md

# =============================================================================
# ## Set segmentation parameters
# =============================================================================

# %% md

# Select an image to test segmentation parameters

# %% codecell
test_index = 2
sn = sample_names[test_index]
input = ip.load_output_file(config, 'raw_fmt',sn)
print(sn)
print(input.shape)

# %% codecell
n_channels = input.shape[2]
n_channels

# %% codecell
clims = [(0,0.1),(0,0.01)] # One color threshold tuple per channel

im_list = [input[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% md

# Subset the image for faster results during parameter selection.

# %% codecell
sr = [(2000,3000),(5000, 6000)]
input_sub = input[sr[0][0]:sr[0][1],sr[1][0]:sr[1][1]]
im_list = [input_sub[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% md

# ### Cell seg

# Get cell seg channels.

# %% codecell
im_cell_list = [input[:,:,i] for i in config['cell_seg']['channels']]
# im_cell_list = [input_sub[:,:,i] for i in config['cell_seg']['channels']]
if len(im_cell_list) > 1:
    im_cell = np.max((np.dstack(im_cell_list)), axis=2)
else:
    im_cell = im_cell_list[0]
im_cell.shape

# %% md

# Test cell seg parameters, adjust in the configuration file, and repeat until it's good. Also repeat with differnt subset regions.

# %% codecell
# need initial values to compare seg changes
im_cell_mask, im_cell_pre, im_cell_seg = [sf.np.zeros((2,2))]*3
im_cell_seg[0,0] = 1

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['cell_seg']

im_cell_mask_old = im_cell_mask

im_cell_mask = sf.get_background_mask(
    im_cell,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold'],
    bg_file=0
    )

# %% codecell
clims=(0,0.1)
print('old')
ip.subplot_square_images([im_cell, im_cell_mask_old], (1,2), clims=[clims,[]],)
ip.plt.show()
print('new')
ip.subplot_square_images([im_cell, im_cell_mask], (1,2), clims=[clims,[]])

# %% codecell
zc = [2500, 0]
zs = [1500, 1000]
clims=(0,0.1)
fig, ax, cbar = ip.plot_image(im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno', clims=clims)

# %% codecell
# config_fn = 'config_mgefish.yaml' # relative path to config file from workdir
config_fn = 'config_mgefish_ecoli.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['cell_seg']

im_cell_pre_old = im_cell_pre
im_cell_seg_old = im_cell_seg

im_cell_pre = sf.pre_process(
    im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
    log=pdict['pre_log'],
    denoise=pdict['pre_denoise'],
    gauss=pdict['pre_gauss'],
    diff_gauss=eval(pdict['diff_gauss'])
    )

im_cell_seg = sf.segment(
    im_cell_pre,
    background_mask = im_cell_mask[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
    n_clust=pdict['n_clust'],
    small_objects=pdict['small_objects']
    )

# %% codecell
clims = ((),(0,0.1), (0,0.5), ())
print('old')
im_list = [im_cell_mask[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
           im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_cell_pre_old,
           ip.seg2rgb(im_cell_seg_old)]
ip.subplot_square_images(im_list, (1,4), clims=clims)
ip.plt.show()
print('new')
im_list = [ im_cell_mask[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
            im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_cell_pre,
            ip.seg2rgb(im_cell_seg)]
ip.subplot_square_images(im_list, (1,4), clims=clims)
# seg_clims = max([clims[i] for i in config['cell_seg']['channels']])
# ip.subplot_square_images(im_list, (1,4), clims=(seg_clims, seg_clims,'',''))


# %% md

# Images above from left to right: Raw, pre-processed, background mask, final segmentation.

# Below the segmentation projected onto the raw image

# %% codecell
clims=(0,0.025)
fig, ax, cbar = ip.plot_image(im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno')
ip.plt.show()
ip.plt.close()
fig, ax, cbar = ip.plot_image(im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno')
ip.plot_seg_outline(ax, im_cell_seg, col=(0,0.75,0))

# %% md

# ### Spot seg

# Test spot seg parameters, adjust in the configuration file, and repeat until it's good. Check all the spot seg channels.

# %% codecell
sp=0
s_ch = config['spot_seg']['channels'][sp]
im_spot = input[:,:,s_ch]

im_spot_mask, im_spot_pre, im_spot_seg = [sf.np.zeros(im_spot.shape)]*3
for i in [im_spot_mask, im_spot_pre, im_spot_seg]:
    i[0,0] = 1

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['spot_seg']

im_spot_mask_old = im_spot_mask

bg_file_fmt = config['output_dir'] + '/' + config['spot_mask_bg_debris_fmt']
bg_file = bg_file_fmt.format(sample_name=sn, spot_chan=s_ch)

im_spot_mask = sf.get_background_mask(
    im_spot,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold'],
    bg_file=bg_file
    )


# %% codecell
clims=(0,0.005)

print('old')
ip.subplot_square_images([im_spot, im_spot_mask_old], (1,2), clims=[clims,[]])
ip.plt.show()
print('new')
ip.subplot_square_images([im_spot, im_spot_mask], (1,2), clims=[clims,[]])

# %% codecell
zc = [2000, 5000]
zs = [1000, 1000]
clims=(0,0.01)
fig, ax, cbar = ip.plot_image(im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno', clims=clims)

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['spot_seg']

im_spot_pre_old = im_spot_pre
im_spot_seg_old = im_spot_seg

im_spot_pre = sf.pre_process(
    im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
    log=pdict['pre_log'],
    denoise=pdict['pre_denoise'],
    gauss=pdict['pre_gauss'],
    diff_gauss=eval(pdict['diff_gauss'])
    )

im_spot_seg = sf.segment(
    im_spot_pre,
    background_mask = im_spot_mask[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
    n_clust=pdict['n_clust'],
    small_objects=pdict['small_objects']
    )

# %% codecell
clims=(0,0.01)
print('old')
im_list = [im_spot_mask_old[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
           im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_spot_pre_old,
           ip.seg2rgb(im_spot_seg_old)]
ip.subplot_square_images(im_list, (1,4), clims=[[],clims,clims,[]])
ip.plt.show()
print('new')
im_list = [ im_spot_mask[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
            im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_spot_pre,
            ip.seg2rgb(im_spot_seg)]
ip.subplot_square_images(im_list, (1,4), clims=[[],clims,clims,[]])
# seg_clims = max([clims[i] for i in config['cell_seg']['channels']])
# ip.subplot_square_images(im_list, (1,4), clims=(seg_clims, seg_clims,'',''))

# %% md

# Images above from left to right: Raw, pre-processed, background mask, final segmentation.

# Below the segmentation projected onto the raw image

# %% codecell
dims=10
fig, ax, cbar = ip.plot_image(im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno', clims=clims, im_inches=dims)
ip.plot_seg_outline(ax, im_spot_seg, col=(0,1,0))

# %% md

# Adjust the local maxima minimum distance parameter. Segmentations with more than one local maximum will be split into different segmentations

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# maxs = spf.peak_local_max(im_spot, min_distance = config['local_max_mindist'])
im_spot_ = im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]]
im_spot_seg_ = (im_spot_seg > 0)
ma = spf._get_merged_peaks(im_spot_ * im_spot_seg_,
                            min_distance=config['local_max_mindist'])
# is_peak = spf.peak_local_max(im_spot, indices=False, min_distance=config['local_max_mindist']) # outputs bool image
# is_peak.shape
# labels = spf.label(is_peak)[0]
# merged_peaks = spf.center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
# ma = np.array(merged_peaks)
fig, ax, cbar = ip.plot_image(im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
                              cmap='inferno', im_inches=20, clims=clims)
ax.scatter(ma[:,1],ma[:,0], s=75, color=(0,1,0))
ax = ip.plot_seg_outline(ax, im_spot_seg, col=(0,0.5,0.5))
# ax.set_xlim((300,400))
# ax.set_ylim((425,475))

# %% md
# =============================================================================
# ## Run the pipeline
# =============================================================================

# %% md

# Write the snakemake execution code to a bash script.

# %% codecell
config_fn = 'config_mgefish.yaml' # relative path to config file from workdir

dry_run = False  # Just create DAG if True
n_cores = 3  # number of allowed cores for the snakemake to use
force_run = False  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['snakefile_segment']
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

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5c/MGEFISH
# $ sh run_Snakefile_segment.sh
# ```

# %% codecell
# E coli configuration
config_fn = 'config_mgefish_ecoli.yaml' # relative path to config file from workdir

dry_run = False  # Just create DAG if True
n_cores = 3  # number of allowed cores for the snakemake to use
force_run = False  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['snakefile_segment']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
command = " ".join(['snakemake', '-s', snakefile, '--configfile', config_fn, '-j',
                    str(n_cores), dr, fr])

run_fn = 'run_{}_ecoli.sh'.format(snakefile)
with open(run_fn, 'w') as f:
    f.write(command)

print(run_fn)
print('$ ', command)

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5c/MGEFISH
# $ sh run_Snakefile_segment_ecoli.sh
# ```

# Check the segmentation

# %% codecell
zf = [0.3,0.5]  # location of the zoom image as a fraction of the total axis len
zs = 2000  # size of the zoom box
for sn in sample_names:
    print(sn)
    _raw_npy_fn = config['output_dir'] + '/raw_npy/{sn}.npy'.format(sn=sn)
    _raw = sf.np.load(_raw_npy_fn)
    _r, _c = round(_raw.shape[0]*zf[0]), round(_raw.shape[1]*zf[1])
    _zc = [_r,_r+zs, _c, _c+zs]
    for ch in config['cell_seg']['channels']:
        print('Cell channel ',ch)
        _cell_seg_fn = (config['output_dir'] + '/cell_seg/{sn}/{sn}_chan_{ch}_cell_seg.npy'.format(sn=sn, ch=ch))
        _raw_chan = _raw[:,:,ch][_zc[0]:_zc[1],_zc[2]:_zc[3]]
        _seg = sf.np.load(_cell_seg_fn)[_zc[0]:_zc[1],_zc[2]:_zc[3]]
        ip.subplot_square_images([_raw_chan, ip.seg2rgb(_seg)], (1,2))
        ip.plt.show()
        ip.plt.close()
    for ch in config['spot_seg']['channels']:
        print('Spot channel ', ch)
        _cell_seg_fn = (config['output_dir'] + '/spot_seg/{sn}/{sn}_chan_{ch}_spot_seg.npy'.format(sn=sn, ch=ch))
        _raw_chan = _raw[:,:,ch][_zc[0]:_zc[1],_zc[2]:_zc[3]]
        _seg = sf.np.load(_cell_seg_fn)[_zc[0]:_zc[1],_zc[2]:_zc[3]]
        ip.subplot_square_images([_raw_chan, ip.seg2rgb(_seg)], (1,2))
        ip.plt.show()
        ip.plt.close()




# %% md
# =============================================================================
# ## Plot SNR curves
# =============================================================================

# Plot pixels normalized by cell count vs snr threshold value

# Get threshold curves

# %% codecell
slims = (0,150)
n_sthr = 300
sthreshs = sf.np.linspace(slims[0], slims[1], n_sthr)
c_ch = config['cell_seg']['channels'][0]

curves_df = pd.DataFrame([])
for sn in sample_names:
    # Get raw image
    raw = ip.load_output_file(config, 'raw_fmt', sn)
    # Get proper channel
    raw_ch = raw[:,:,s_ch]
    for s_ch in config['spot_seg']['channels']:
        # Get background value
        bg_mask = ip.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=s_ch)
        bg = raw_ch[bg_mask == 0]
        bg_val = sf.np.mean(bg)
        # Get SNR
        props = ip.load_output_file(
                config, 'spot_props_max_split_fmt', sn, spot_chan=s_ch
                )
        props['snr'] = props.max_intensity / bg_val
        # Get the number of cells
        cprops = ip.load_output_file(config, 'cell_props_fmt', sn, cell_chan=c_ch)
        n = cprops.shape[0]
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
spot_curve_bg_fmt = config['output_dir'] + '/fig_5b_spot_count_curve_snr_thresh'
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
ip.plt.show()
