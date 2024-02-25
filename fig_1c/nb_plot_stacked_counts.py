# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created : 5/20/22
# =============================================================================
"""
The notebook Has Been Built for...Running the segmentation_pipeline

For use with the 'hiprfish_imaging_py38' conda env
"""
# %% codecell
# =============================================================================
# Setup
# =============================================================================
# Modify
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/phage/experiments/2022_05_06_ecoliphagegel'
                    # Absolute path to the project work directory



# %% codecell
# Imports
import glob
import pandas as pd
import numpy as np
import gc
import os
import re
import sys

# %% codecell
# Set up notebook stuff
%load_ext autoreload
%autoreload 2
gc.enable()

# %% codecell
# Move to notebook directory
os.chdir(project_workdir)
os.getcwd()

# %% codecell
# other imports
sys.path.append('/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code')
import image_plots as ip
import image_functions as ifn

# %% codecell
# =============================================================================
# Setup
# =============================================================================
#  get input files automatically
input_filenames = glob.glob('data/CellCounter_*.csv')
input_filenames.sort()
print(len(input_filenames))
input_filenames

# %% codecell
# get values
count_dfs = []
for fn in input_filenames:
    count_dfs.append(pd.read_csv(fn)[['Type 1','Type 2','Type 3']])
count_dfs[0]

# %% codecell
# get fractions
fracs = []
for c in count_dfs:
    total = c.sum(axis=1).values[0]
    f = c.div(total)
    f['total'] = total
    fracs.append(f)
fracs

# %% codecell
# get fn keys
factors = ['moi','time']
keys = [ifn.get_filename_keys(sn, factors) for sn in input_filenames]
keys

# %% codecell
# group fractions
mois = ['000','001','010','100']
times = ['010','020','030','040']
frac_g1,frac_g2,frac_g3,frac_g4 = [],[],[],[]
for moi in mois:
    for time in times:
        for k, f in zip(keys,fracs):
            if (k[0] == moi) & (k[1] == time):
                frac_g1.append(f['Type 1'].values[0])
                frac_g2.append(f['Type 2'].values[0])
                frac_g3.append(f['Type 3'].values[0])
frac_g1 = np.array(frac_g1)
frac_g2 = np.array(frac_g2)
frac_g3 = np.array(frac_g3)

# %% codecell
# Plot
ft=(7)
dims=(1.8307, 1.9685)
color='k'
lw=1
w=0.7

fig, ax = ip.general_plot(col=color, dims=dims, ft=ft, lw=lw)
x = [1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]
# x = [0.7,0.9,1.1,1.3,1.8,1.9,2,2.1,2.8,2.9,3,3.1,3.8,3.9,4,4.1]
ax.bar(x, frac_g3, label='count ' + r'$\geq$' + ' 5', color=(1,0,1,1))
ax.bar(x, frac_g2, bottom=frac_g3, label='count < 5', color=(1,0,1,0.25))
ax.bar(x, frac_g1, bottom=frac_g2 + frac_g3, label='count = 0', color=(0,1,1))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=ft*5/7)
ax.tick_params(axis='x', which='both',bottom=False, top=False, labelbottom=False)
# ax.set_ylim(0, 0.25)
# ax.set_xticks(x, minor=True)
# minor_labels = np.tile(['10','20'], 4)
# # ax.set_xticklabels(minor_labels)
# ax.set_xticks([1.5, 4.5, 7.5, 10.5])
# major_labels = ['0','0.01','0.1','1']
# ax.set_xticklabels(major_labels)

fig_dir = 'figures'
out_fname = '{}/count_fractions_stacked_bar'.format(fig_dir)
ip.save_png_pdf(out_fname)
