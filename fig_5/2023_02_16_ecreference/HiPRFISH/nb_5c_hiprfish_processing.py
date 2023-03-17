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

# %% md

# Move to the workdir

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2023_02_16_ecreference/HiPRFISH'

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

# %% md

# ==============================================================================
# ## Reference training
# ==============================================================================

# Train the spectral classifier using the hiprfish probe design file

# %% codecell
ref_train_out_dir = (
                    config['output_dir']
                    + '/' + config['reference_training']['out_dir']
                    )

fgu.check_dirs(ref_train_out_dir)

classifier_script = (
        config['__default__']['SCRIPTS_PATH'] + '/'
        + config['reference_training']['script']
        )

command = " ".join(['python', classifier_script, '-c', config_fn])

run_fn = 'run_{}.sh'.format('hiprfish_training')
with open(run_fn, 'w') as f:
    f.write(command)


command

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2022_12_06_restain
# $ sh run_hiprfish_training.sh
# ```

# %% md

# ==============================================================================
# ## Prepare image data for analysis
# ==============================================================================

# Get sample names

# %% codecell
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

regex = config['__default__']['DATA_DIR'] + '/*' + config['laser_regex']
print(regex)
input_fns = glob.glob(regex)
input_fns = [os.path.split(fn)[1] for fn in input_fns]
input_fns = [re.sub(config['laser_regex'], '', fn) for fn in input_fns]
sample_names = np.unique(input_fns).tolist()
sample_names


# %% md

# ### Construct the input table

# Notes on column names: SAMPLE really means the sample directory path

# IMAGES are the sample names that go with each set of spectral images: 'IMAGE_488', 'IMAGE_514'...etc.

# CALIBRATION_FILENAME is for a flat field image correction

# REFERENCE_FOLDER is the path to the trained classifier with 'SPC' number of simulations

# DIMENSION is the number of image dimensions (i.e. a z stack is "3")

# ZSLICE specifies only one 2D image from a z stack, set to '-1' to set this feature off

# PROBEDESIGN is the probe design output filename used in the reference training

# SIZELIMIT is the minimum size of allowed objects in the segmentation

# %% codecell
column_names = [
        'SAMPLE','IMAGES','SAMPLING_TIME','SAMPLING_DAY_REFERENCE','MOUSE',
        'TREATMENT','CALIBRATION','REFERENCE','CALIBRATION_FILENAME',
        'REFERENCE_FOLDER','SPC','DIMENSION','ZSLICE','PROBEDESIGN',
        'DESIGN_ID','SIZELIMIT','MARKER_X','MARKER_Y'
        ]
input_table = pd.DataFrame(columns=column_names)

input_table['IMAGES'] = sample_names
input_table['SAMPLE'] = '.'
input_table['SAMPLING_TIME'] = 0
input_table['SAMPLING_DAY_REFERENCE'] = 0
input_table['MOUSE'] = 0
input_table['TREATMENT'] = 0
input_table['CALIBRATION'] = 'F'
input_table['REFERENCE'] = 'T'
input_table['CALIBRATION_FILENAME'] = 0
input_table['REFERENCE_FOLDER'] = config['reference_training']['out_dir']
input_table['CALIBRATION_FILENAME'] = 0
input_table['SPC'] = 2000
input_table['DIMENSION'] = 2
input_table['ZSLICE'] = -1
input_table['PROBEDESIGN'] = (config['__default__']['PROBE_DESIGN_DIR']
                                + '/' + config['probe_design_filename'])
input_table['DESIGN_ID'] = config['probe_design_id']
input_table['SIZELIMIT'] = 200
input_table['MARKER_X'] = 0
input_table['MARKER_Y'] = 0
input_table.to_csv(config['images']['image_list_table'], index=False)
input_table

# %% md

# ==============================================================================
# ## Run segment script
# ==============================================================================

# Write the snakemake execution code to a bash script.

# %% codecell
dry_run = True  # Just create DAG if True
n_cores = 23  # number of allowed cores for the snakemake to use
force_run = 'classify_spectra'  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['segmentation']['snakefile']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
r = '--resources mem_gb=' + str(config['resource_mem_gb'])
command = " ".join(['snakemake', '-s', snakefile, '--configfile', config_fn, '-j',
                    str(n_cores), r, dr, fr])

run_fn = 'run_{}.sh'.format(snakefile)
with open(run_fn, 'w') as f:
    f.write(command)

print(run_fn)
print('$ ', command)

# %% md

# ==============================================================================
# ## Choose Seg parameters
# ==============================================================================

# Select an image to test segmentation parameters

# %% codecell
test_index = 0
sn = sample_names[test_index]
input = ip.load_output_file(config, 'sum_fmt',sn)
print(sn)
print(input.shape)


# %% codecell
clims = [] # One color threshold tuple per channel
n_channels=1
im_list = [input]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)

# %% md

# Subset the image for faster results during parameter selection.

# %% codecell
sr = [(1000,1500),(1000, 1500)]
input_sub = input[sr[0][0]:sr[0][1],sr[1][0]:sr[1][1]]
input_sub.shape
im_list = [input_sub]
ip.subplot_square_images(im_list, (1,n_channels), clims=clims)


# %% md

# Test cell seg parameters, adjust in the configuration file, and repeat until it's good. Also repeat with differnt subset regions.

# %% codecell
# need initial values to compare seg changes
im_cell = input
im_cell_mask, im_cell_pre, im_cell_seg = [sf.np.zeros((2,2))]*3
im_cell_seg[0,0] = 1

# %% codecell
# Generate Background mask
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
# View background bg_mask
clims=()
print('old')
ip.subplot_square_images([im_cell, im_cell_mask_old], (1,2), clims=[clims,[]],)
ip.plt.show()
print('new')
ip.subplot_square_images([im_cell, im_cell_mask], (1,2), clims=[clims,[]])

# %% md

# Run segmentation on a zoom region

# %% codecell
# Pick zoom region
zc = [1400, 800]
zs = [500, 500]
clims=('min','max')
fig, ax, cbar = ip.plot_image(im_cell[zc[0]:zc[0]+zs[0],zc[1]:zc[1]+zs[1]], cmap='inferno', clims=clims)

# %% codecell
# Run segmentation
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
    window=pdict['window'],
    n_clust=pdict['n_clust'],
    small_objects=pdict['small_objects']
    )

# %% codecell
# Check segmentation
clims = ((),(), (), ())
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

# Run segmentation-classification pipeline

# %% codecell
dry_run = False  # Just create DAG if True
n_cores = 1  # number of allowed cores for the snakemake to use
force_run = False  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['snakefilename']
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

# ==============================================================================
# ## Run segment script
# ==============================================================================

# Write the snakemake execution code to a bash script.

# %% codecell
dry_run = False  # Just create DAG if True
n_cores = 21  # number of allowed cores for the snakemake to use
force_run = False  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['segmentation']['snakefile']
dr = '-pn' if dry_run else '-p'
fr = '-R ' + force_run if force_run else ''
r = '--resources mem_gb=' + config['resource_mem_gb']
command = " ".join(['snakemake', '-s', snakefile, '--configfile', config_fn, '-j',
                    str(n_cores), r, dr, fr])

run_fn = 'run_{}.sh'.format(snakefile)
with open(run_fn, 'w') as f:
    f.write(command)

print(run_fn)
print('$ ', command)













# %% md

# ### Run image prep

# Write the snakemake execution code to a bash script.

# %% codecell
script_fn = (
        config['__default__']['SCRIPTS_PATH'] + '/'
        + config['prep']['script']
        )

command = " ".join(['python', script_fn, config_fn])

run_fn = 'run_{}.sh'.format('prep')
with open(run_fn, 'w') as f:
    f.write(command)

print(run_fn)
print('$ ', command)

# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd mgefish/code/fig_5/fig_5c/HiPRFISH
# $ sh run_prep.sh
# ```

# %% md

a = 0
a = np.array([1,2,3])
b = np.ones((3,3))
c = b*a
c
d = np.array([[1,2,3],[4,5,6],[7,8,9]])
e = b*d

plt.plot(c.T, e.T, color='k', alpha=0.5, lw=0.5)

(np.ones((2,2,3)) * [1,2,3])[:,:,2]

im = np.array([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0],[0,0,0,2,2],[0,0,0,2,2]])

dict_label_bbox = {1:(0,0,2,2), 2:(3,3,5,5)}

dict_label_alt = {1:[1,1,0], 2: [0,0,2]}

def recolor_image(im, dict_label_bbox, dict_label_alt, threeD=0):
    shape = im.shape + (threeD,) if threeD else im.shape
    im_alt = np.zeros(shape)
    for label, bbox in dict_label_bbox.items():
        box = (im[bbox[0]:bbox[2],bbox[1]:bbox[3]] == label)*1
        box = box[...,None] if threeD else box
        alt = dict_label_alt[label]
        box_alt = (box * alt) + im_alt[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        im_alt[bbox[0]:bbox[2],bbox[1]:bbox[3]] = box_alt
    return(im_alt)

recolor_image(im, dict_label_bbox, dict_label_alt, threeD=3)

a = np.array([[1,0],[0,1]])

(a[...,None] * np.array([2,3]))[:,:,1]


# %% md

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5b/HiPRFISH
# $ sh run_hiprfish_training.sh
# ```
