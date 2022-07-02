# %% md

# # Figure 3b: Image Segmentation

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
%load_ext autoreload
%autoreload 2

sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip
import segmentation_func as sf
import spot_funcs as spf

# %% md

# =============================================================================
# ## Set segmentation parameters
# =============================================================================

# %% md

# Get the input filenames. Reload the config file if you need to update the regular expression.

# %% codecell
input_filenames = glob.glob(config['input_dir'] + '/' + config['input_regex']
                            + config['input_ext'])
input_filenames.sort()
print(len(input_filenames))
input_filenames

# %% md

# Select an image to test segmentation parameters

# %% codecell
test_index = 1
input_fn = input_filenames[test_index]
input_fn

# %% codecell
javabridge.start_vm(class_path=bioformats.JARS)
input = bioformats.load_image(input_fn)
input.shape

# %% codecell
n_channels = input.shape[2]
n_channels

# %% codecell
clims = [(0,0.05),(0,0.05),(0,0.05),(0,0.25)] # One color threshold tuple per channel
im_list = [input[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% md

# Subset the image for faster results during parameter selection.

# %% codecell
sr = [(2000,2500),(2000, 2500)]
input_sub = input[sr[0][0]:sr[0][1],sr[1][0]:sr[1][1]]
im_list = [input_sub[:,:,i] for i in range(n_channels)]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% md

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

# Test spot seg parameters, adjust in the configuration file, and repeat until it's good. Also repeat with differnt subset regions.

# %% codecell
im_cell_mask_old, im_cell_pre_old, im_cell_seg_old = [sf.np.zeros((2,2))]*3
im_cell_seg_old[0,0] = 1

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['cell_seg']

im_cell_mask = sf.get_background_mask(
    im_cell,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold']
    )
print('old')
ip.subplot_square_images([im_cell, im_cell_mask_old], (1,2))
ip.plt.show()
print('new')
ip.subplot_square_images([im_cell, im_cell_mask], (1,2))
im_cell_mask_old = im_cell_mask

# %% codecell
zc = [2000, 2000]
zs = [500, 500]
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['cell_seg']

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
print('old')
im_list = [im_cell_mask_old[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
           im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_cell_pre_old,
           ip.seg2rgb(im_cell_seg_old)]
ip.subplot_square_images(im_list, (1,4))
ip.plt.show()
print('new')
im_list = [ im_cell_mask[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
            im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_cell_pre,
            ip.seg2rgb(im_cell_seg)]
ip.subplot_square_images(im_list, (1,4))
im_cell_pre_old = im_cell_pre
im_cell_seg_old = im_cell_seg
# seg_clims = max([clims[i] for i in config['cell_seg']['channels']])
# ip.subplot_square_images(im_list, (1,4), clims=(seg_clims, seg_clims,'',''))

# %% md

# Images above from left to right: Raw, pre-processed, background mask, final segmentation.

# Below the segmentation projected onto the raw image

# %% codecell
fig, ax, cbar = ip.plot_image(im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno')
ip.plot_seg_outline(ax, im_cell_seg, col=(0,0,0))

# %% md

# Test spot seg parameters, adjust in the configuration file, and repeat until it's good. Check all the spot seg channels.

# %% codecell
sp=0
im_spot_list = [input[:,:,i] for i in config['spot_seg']['channels']]
im_spot = im_spot_list[sp]

# %% codecell
im_spot_mask_old, im_spot_pre_old, im_spot_seg_old = [sf.np.zeros((2,2))]*3
im_spot_seg_old[0,0] = 1

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['spot_seg']

im_spot_mask = sf.get_background_mask(
    im_spot,
    bg_filter=pdict['bg_filter'],
    bg_log=pdict['bg_log'],
    bg_smoothing=pdict['bg_smoothing'],
    n_clust_bg=pdict['n_clust_bg'],
    top_n_clust_bg=pdict['top_n_clust_bg'],
    bg_threshold=pdict['bg_threshold']
    )
print('old')
ip.subplot_square_images([im_spot, im_spot_mask_old], (1,2))
ip.plt.show()
print('new')
ip.subplot_square_images([im_spot, im_spot_mask], (1,2))
im_spot_mask_old = im_spot_mask

# %% codecell
zc = [4000, 4000]
zs = [500, 500]
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)
pdict = config['spot_seg']

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
print('old')
im_list = [im_spot_mask_old[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
           im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_spot_pre_old,
           ip.seg2rgb(im_spot_seg_old)]
ip.subplot_square_images(im_list, (1,4))
ip.plt.show()
print('new')
im_list = [ im_spot_mask[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
            im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], im_spot_pre,
            ip.seg2rgb(im_spot_seg)]
ip.subplot_square_images(im_list, (1,4))
im_spot_pre_old = im_spot_pre
im_spot_seg_old = im_spot_seg
# seg_clims = max([clims[i] for i in config['cell_seg']['channels']])
# ip.subplot_square_images(im_list, (1,4), clims=(seg_clims, seg_clims,'',''))

# %% md

# Images above from left to right: Raw, pre-processed, background mask, final segmentation.

# Below the segmentation projected onto the raw image

# %% codecell
fig, ax, cbar = ip.plot_image(im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno')
ip.plot_seg_outline(ax, im_spot_seg, col=(0,0,0))

# %% md

# Adjust the local maxima minimum distance parameter. Segmentations with more than one local maximum will be split into different segmentations

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# maxs = spf.peak_local_max(im_spot, min_distance = config['local_max_mindist'])
ma = spf._get_merged_peaks(im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
                            min_distance=config['local_max_mindist'])
# is_peak = spf.peak_local_max(im_spot, indices=False, min_distance=config['local_max_mindist']) # outputs bool image
# is_peak.shape
# labels = spf.label(is_peak)[0]
# merged_peaks = spf.center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
# ma = np.array(merged_peaks)
fig, ax, cbar = ip.plot_image(im_spot[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]],
                              cmap='inferno', im_inches=20)
ax.scatter(ma[:,1],ma[:,0], s=50, color=(0,1,0))
ax = ip.plot_seg_outline(ax, im_spot_seg, col=(0,0.8,0.8))
# ax.set_xlim((300,400))
# ax.set_ylim((425,475))

# %% md
# =============================================================================
# ## Run the pipeline
# =============================================================================

# To run a test of the pipeline write an input table with only one file.

# %% codecell
input_fns_split = [os.path.split(fn)[1] for fn in [input_fn]]
sample_names = [re.sub(config['input_ext'], '', fn) for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['input_table_fn'], index=False)
sample_names

# %% md

# To run all the samples write a full input table.

# %% codecell
input_fns_split = [os.path.split(fn)[1] for fn in input_filenames]
sample_names = [re.sub(config['input_ext'], '', fn) for fn in input_fns_split]
input_table = pd.DataFrame(sample_names, columns=config['input_table_cols'])
input_table.to_csv(config['input_table_fn'], index=False)
input_table.values

# %% md

# Write the snakemake execution code to a bash script.

# %% codecell
dry_run = False  # Just create DAG if True
n_cores = 2  # number of allowed cores for the snakemake to use
force_run = 'segment_cells'  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['snakefile']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
command = " ".join(['snakemake', '-s', snakefile, '--configfile', config_fn, '-j',
                    str(n_cores), dr, fr])

with open(config['run_fn'], 'w') as f:
    f.write(command)

command

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3b
# $ sh run.sh
# ```

# Check the segmentation

# %% codecell
zf = [0.5,0.4]  # location of the zoom image as a fraction of the total axis len
zs = 1000  # size of the zoom box
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
