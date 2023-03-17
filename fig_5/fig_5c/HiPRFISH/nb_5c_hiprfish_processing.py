# %% md

# # Figure 5b: HiPRFISH processing

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
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5c/HiPRFISH'

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
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5c/HiPRFISH
# $ sh run_hiprfish_training.sh
# ```

# %% md

# ==============================================================================
# ## Prepare image data for analysis
# ==============================================================================

# Get sample names

# %% codecell
input_fns = glob.glob(config['__default__']['DATA_DIR'] + '/*spec*.czi')
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

# Now execute the script in the command line.

# ```console
# $ conda activate hiprfish_imaging_py38
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/fig_5b/HiPRFISH
# $ sh run_hiprfish_training.sh
# ```
