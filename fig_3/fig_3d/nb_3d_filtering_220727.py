# %% md

# # Figure 3b: Segmentation filtering

# Ran after running nb_3b_segmentation.py. Used the "esda" conda environment.

# %% md
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

# %% md

# Move to the workdir

# %% codecell
# Absolute path
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3d'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory

# %% md

# Load all the variables from the segmentation pipeline

# %% codecell
config_fn = 'config.yaml' # relative path to config file from workdir

with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

# %% md

# Specify an output directory for plots

# %% codecell

fig_dir = config['figure_dir']
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    print("made dir:", fig_dir)

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%load_ext autoreload
%autoreload 2

sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import fn_analysis_plots as apl

# %% md

# Get sample names

# %% codecell
sample_names = pd.read_csv(config['input_table_fn']).sample_name.values
sample_names

# %% md
# ==============================================================================
# ## Area thresholding on the cell segmentation
# ==============================================================================

# ### Remove off-target binding to debris in segmentations by area thresholding

# Plot number of cells as a function of area threshold value and select threshold values for each channel. For the lower threshold, zoomed in to the left part of the x axis on the threshold curves. For the upper threshold tried to remove only the very largest objects, Checked visually on teh segmentation.

# %% codecell
# Get cell area threshold curves
t_lims = (0,10000)
n = 100
fmt = 'cell_props_fmt'
vals = 'area'
seg_type = 'cell_seg'
threshs = np.linspace(t_lims[0], t_lims[1], n)
cell_area_curve_dict = apl.get_curve_dict(sample_names, config, seg_type=seg_type,
                                        fmt=fmt, vals=vals,
                                        threshs=threshs, lessthan=True)
# %% codecell
# Plot curves
xlims=(0,10000)
dims=(4, 2)
lw=1
ft=6
dpi=1000
labels = ['a']
threshold_picks = [[75,7500]]
for sn in sample_names:
    for ch, th_pck in zip(config[seg_type]['channels'], threshold_picks):
        curves = [cell_area_curve_dict[sn][ch]]
        fig, ax = apl.plot_threshold_curves(threshs, curves, xlims=xlims,
                                            dims=dims, lw=lw, ft=ft)
        # for c in curves:
        #     slope = apl.get_curve_slope(threshs.tolist(), c)
        #     ax.plot(threshs, slope/np.max(slope))
        for t in th_pck:
            ax.plot([t]*2, [0,1], 'k', lw=lw)
        output_basename = (fig_dir + '/' + sn + '_chan_' + str(ch)
                            + '_cell_area_threshold_curves')
        apl.save_png_pdf(output_basename)

# %% md

# Add boolean area thresholding column to cell segmentation properties tables

# %% codecell
# Add to cell props
# cell_props_area_fmt = re.sub('.csv', '_area_filter.csv', (config['output_dir']
#                              + '/' + config[fmt]))
cell_props_area_fmt = config['output_dir'] + '/' + config['cell_props_area_filt_fmt']
for sn in sample_names:
    for ch, th_pck in zip(config[seg_type]['channels'], threshold_picks):
        props = apl.load_output_file(config, 'cell_props_fmt', sn, ch)
        ut = (props.area < th_pck[1]).astype(int)
        lt = (props.area > th_pck[0]).astype(int)
        props['area_thresh'] = ut * lt
        output_filename = cell_props_area_fmt.format(sample_name=sn,
                                                       cell_chan=ch)
        props.to_csv(output_filename, index=False)
        print(output_filename)


# %% md

# Show thresholded images

# %% codecell
zc = (2500, 4500, 4000, 6000)
clims=(0,0.05)
cell_seg_area_filt_fmt = config['output_dir'] + '/' + config['cell_seg_area_filt_fmt']
for sn in sample_names:
    for ch, th_pck in zip(config[seg_type]['channels'], threshold_picks):
        raw = apl.load_output_file(config, 'raw_fmt', sn, ch)[:,:,ch]
        seg = apl.load_output_file(config, 'cell_seg_fmt', sn, ch)
        seg_new = seg.copy()
        props = pd.read_csv(cell_props_area_fmt.format(sample_name=sn, cell_chan=ch))
        remove_cids = props.loc[props.area_thresh == 0, ['label','bbox']].values
        for i, (c, b) in enumerate(remove_cids):
            if i == 0:  print('yep')
            b = eval(b)
            b_sub = seg_new[b[0]:b[2],b[1]:b[3]]
            b_sub = b_sub * (b_sub != c)
            seg_new[b[0]:b[2],b[1]:b[3]] = b_sub
        output_filename = cell_seg_area_filt_fmt.format(sample_name=sn,
                                                        cell_chan=ch)
        np.save(output_filename, seg_new)
        apl.plot_image(raw, zoom_coords=zc, cmap='inferno', clims=clims)
        plt.show()
        plt.close()
        apl.plot_image(seg>0, zoom_coords=zc)
        plt.show()
        plt.close()
        apl.plot_image(seg_new>0, zoom_coords=zc)
        plt.show()
        plt.close()




# %% md
# ==============================================================================
# ## SNR thresholding on the spot segmentations
# ==============================================================================

# There are many segmented spots that are only part of the background noise.

# ### Calculate the SNR for spots

# %% codecell
# Get background values and calculate SNR
c_ch = config['cell_seg']['channels'][0]
spot_props_snr_filt_fmt = config['output_dir'] + '/' + config['spot_props_snr_fmt']
bg_dict = {}
for sn in sample_names:
    raw = apl.load_output_file(config, 'raw_fmt', sn)
    bg_dict[sn] = {}
    for ch in config['spot_seg']['channels']:
        # Background
        raw_ch = raw[:,:,ch]
        seg = apl.load_output_file(config, 'spot_mask_bg_rough_fmt', sn, spot_chan=ch)
        bg = raw_ch[seg == 0]
        bg_mean = np.mean(bg)
        bg_std = np.std(bg)
        bg_dict[sn][ch] = [bg_mean, bg_std]
        # SNR
        props = apl.load_output_file(config, 'spot_props_max_split_fmt', sn, spot_chan=ch,
                                     cell_chan=c_ch)
        props['snr'] = props.max_intensity / bg_mean
        output_filename = spot_props_snr_filt_fmt.format(sample_name=sn, spot_chan=ch)
        props.to_csv(output_filename, index=False)


# %% codecell
# PLot background values
dims=(2,2)
ticks=[]
tab10 = apl.get_cmap_listed('tab10')
cols = tab10[:6]
fig, ax = apl.general_plot(dims=dims)
for k, (sn, c) in enumerate(zip(sample_names, cols)):
    print(sn)
    shift = k
    ticks.append(shift+0.2)
    for l, ch in enumerate(config['spot_seg']['channels']):
        bgm = bg_dict[sn][ch][0]
        bgs = bg_dict[sn][ch][1]
        shift2 = shift + l*0.2
        ax.errorbar(shift2, bgm, yerr=bgs, marker='.', color=c)

ax.set_xticks(ticks)
ax.set_xticklabels([])

# %% md

# ### Remove background spots by SNR thresholding

# Plot number of objects as a function of SNR threshold value and select a threshold for each channel.

# %% codecell
for sn in sample_names:
    for ch in config['spot_seg']['channels']:
        props = apl.load_output_file(config, 'spot_props_snr_fmt', sn, spot_chan=ch,
                                     cell_chan=c_ch)
        print(props.snr.min(), props.snr.mean(), props.snr.max())

# %% codecell
t_lims = (0,100)
n = 1000
vals = 'snr'
seg_type = 'spot_seg'
threshs = np.linspace(t_lims[0], t_lims[1], n)
spot_snr_curve_dict = {}
for sn in sample_names:
    spot_snr_curve_dict[sn] = {}
    for ch in config['spot_seg']['channels']:
        props = apl.load_output_file(config, 'spot_props_snr_fmt', sn, spot_chan=ch,
                                     cell_chan=c_ch)
        # props = props[props.cell_id > 0]
        n = props.shape[0]
        spot_snr_curve_dict[sn][ch] = [props[props[vals] > t].shape[0] / n
                                        for t in threshs]

# %% md

# Picked a threshold after the initial steep negative slope, where the number of spots becomes less sensitive to the threshold.

# %% codecell
# Plot curves
xlims=(0,100)
dims=(6,3)
lw=1
ft=6
dpi=1000
threshold_picks = [17]
for ch, thr_pk in zip(config['spot_seg']['channels'], threshold_picks):
    for sn in sample_names:
        print('Sample: ', sn, '\nChannel: ', ch)
        curves = [spot_snr_curve_dict[sn][ch]]
        fig, ax = apl.plot_threshold_curves(threshs, curves, xlims=xlims,
                                            dims=dims, lw=lw, ft=ft)
        # for c in curves:
        #     slope = apl.get_curve_slope(threshs.tolist(), c)
        #     ax.plot(threshs, slope/np.max(slope))
        ax.plot([thr_pk]*2, [0,1], 'k', lw=lw)
        ax.plot(t_lims, [0,0],'k',lw=lw*0.5)
        output_basename = (fig_dir + '/' + sn + '_chan_' + str(ch)
                            + '_spot_snr_threshold_curves')
        apl.save_png_pdf(output_basename)
        plt.show()
        plt.close()

# %% md

# Add boolean area thresholding column to spot segmentation properties tables

# %% codecell
# Add to spot props
spot_props_snr_filt_fmt = config['output_dir'] + '/' + config['spot_props_snr_filt_fmt']
for sn in sample_names:
    for ch, thr_pk in zip(config['spot_seg']['channels'], threshold_picks):
        props = apl.load_output_file(config, 'spot_props_snr_fmt', sn, spot_chan=ch,
                                     cell_chan=c_ch)
        props['snr_thresh'] = (props.snr > thr_pk).astype(int)
        output_filename = spot_props_snr_filt_fmt.format(sample_name=sn,
                                                         spot_chan=ch)
        props.to_csv(output_filename, index=False)

# %% md

# Show thresholded images

# %% codecell
zc = (2000, 4000, 2000, 4000)
spot_seg_snr_filt_fmt = config['output_dir'] + '/' + config['spot_seg_snr_filt_fmt']
for sn in sample_names:
    raw = apl.load_output_file(config, 'raw_fmt', sn)
    for ch in config['spot_seg']['channels']:
        print('Sample: ', sn, '\nChannel: ', ch)
        raw_ch = raw[:,:,ch]
        seg = apl.load_output_file(config,'spot_seg_max_split_fmt', sn,
                                    cell_chan=c_ch, spot_chan=ch)
        props = apl.load_output_file(config,'spot_props_snr_filt_fmt', sn, spot_chan=ch)
        seg_new = apl.filter_seg_objects(seg, props, 'snr_thresh')
        output_filename = spot_seg_snr_filt_fmt.format(sample_name=sn,
                                                            spot_chan=ch)
        np.save(output_filename, seg_new)
        print('Pre: ', props.shape[0])
        print('Post (table): ', props[props.snr_thresh==1].shape[0])
        print('Post (seg): ', np.unique(seg_new).shape[0])
        apl.plot_image(raw_ch, zoom_coords=zc, cmap='inferno', im_inches=10)
        plt.show()
        plt.close()
        apl.plot_image(seg>0, zoom_coords=zc, im_inches=10)
        plt.show()
        plt.close()
        apl.plot_image(seg_new>0, zoom_coords=zc, im_inches=10)
        plt.show()
        plt.close()
        fig, ax, cbar = apl.plot_image(raw_ch[zc[0]:zc[1],zc[2]:zc[3]], cmap='inferno', im_inches=10)
        apl.plot_seg_outline(ax, seg[zc[0]:zc[1],zc[2]:zc[3]])
        plt.show()
        plt.close()
        fig, ax, cbar = apl.plot_image(raw_ch[zc[0]:zc[1],zc[2]:zc[3]], cmap='inferno', im_inches=10)
        apl.plot_seg_outline(ax, seg_new[zc[0]:zc[1],zc[2]:zc[3]])
        plt.show()
        plt.close()

# %% md
# ==============================================================================
# ## Assing spots to cells
# ==============================================================================

# Given a spot segmentation and a cell segmentation, determine wich spots associate with which cells.

# Run a snakemake for this

# %% codecell
dry_run = False  # Just create a plan for the run if True
n_cores = 6  # number of allowed cores for the snakemake to use
force_run = 'assign_spots_to_cells_220707'  # Pick a rule to re-run. False if you don't want a force run.

snakefile = config['snakefile_spottocell']
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
# $ cd /fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3b
# $ sh run_Snakefile_spottocell.sh
# ```

# Check spot to cell assignment

# %% codecell
sn = sample_names[0]
s_ch = config['spot_seg']['channels'][0]
c_ch = config['cell_seg']['channels'][0]
zc = (2000,6000, 2000,6000)
cell_seg = apl.load_output_file(config, 'cell_seg_area_filt_fmt', sn, cell_chan=c_ch)
spot_props_snr = apl.load_output_file(config, 'spot_props_snr_filt_fmt', sn,
                                    cell_chan=c_ch, spot_chan=s_ch)
ss = spot_props_snr[spot_props_snr.snr_thresh==1].centroid.apply(lambda x: eval(x))
ss = np.array([[i for i in k] for k in ss])
ss
# ss = np.array([[i - j for i, j in zip(k, [zc[0],zc[2]])] for k in ss])
spot_props_cid = apl.load_output_file(config, 'spot_props_cid_fmt', sn,
                                    cell_chan=c_ch, spot_chan=s_ch)
sc = spot_props_snr[spot_props_cid.cell_id > 0].centroid.apply(lambda x: eval(x))
sc = np.array([[i for i in k] for k in sc])
sc
print('Pre')
fig, ax, cbar = apl.plot_image(cell_seg > 0, cmap='gray', zoom_coords=zc)
ax.plot(ss[:,1], ss[:,0], '.', ms=4, color=(1,0,1))
plt.show()
print('Post')
fig, ax, cbar = apl.plot_image(cell_seg > 0, cmap='gray', zoom_coords=zc)
ax.plot(sc[:,1], sc[:,0], '.', ms=4, color=(1,0,1))
plt.show()
# apl.plot_seg_outline(ax, spot_seg_cid[zc[0]:zc[1],zc[2]:zc[3]], col=(0,1,1))


# %% md
# ==============================================================================
# ## Final checks
# ==============================================================================

# Get a table of values after all the thresholding. Make sure the segmentation matches the properties table

# %% codecell
counts_seg = pd.DataFrame([])
for sn in sample_names:
    counts = []
    index = []
    for c_ch in config['cell_seg']['channels']:
        cell_seg = apl.load_output_file(config, 'cell_seg_fmt', sn, cell_chan=c_ch)
        cp_c = np.unique(cell_seg).shape[0]
        cell_seg_area_filt = apl.load_output_file(config, 'cell_seg_area_filt_fmt', sn, cell_chan=c_ch)
        cpa_c = np.unique(cell_seg_area_filt).shape[0]
        counts += [i-1 for i in [cp_c, cpa_c]]
        index += ['cell_seg','cell_area_filt']
        for s_ch in config['spot_seg']['channels']:
            spot_seg = apl.load_output_file(config, 'spot_seg_fmt', sn, spot_chan=s_ch)
            sp_c = np.unique(spot_seg).shape[0]
            # spot_seg_area_filt = apl.load_output_file(config, 'spot_seg_area_filt_fmt', sn, spot_chan=s_ch)
            # spa_c = np.unique(spot_seg_area_filt).shape[0]
            spot_seg_max_split = apl.load_output_file(config, 'spot_seg_max_split_fmt', sn, spot_chan=s_ch)
            spm_c = np.unique(spot_seg_max_split).shape[0]
            spot_seg_snr_filt = apl.load_output_file(config, 'spot_seg_snr_filt_fmt', sn, spot_chan=s_ch, cell_chan=c_ch)
            spc_c = np.unique(spot_seg_snr_filt).shape[0]
            spot_seg_cid_filt = apl.load_output_file(config, 'spot_seg_cid_filt_fmt', sn, spot_chan=s_ch, cell_chan=c_ch)
            spas_c = np.unique(spot_seg_cid_filt).shape[0]
            counts += [i-1 for i in [sp_c, spm_c, spc_c, spas_c]]
            _ind = ['spot_seg','spot_max_split','spot_snr_filt', 'spot_cid_filt']
            index += [i + '_ch_' + str(s_ch) for i in _ind]
    counts_seg[sn] = counts
    counts_seg.index = index

counts_seg.to_csv(config['output_dir'] + '/counts_seg.csv')
counts_seg

# %% codecell
counts_table = pd.DataFrame([])
for sn in sample_names:
    counts = []
    index = []
    for c_ch in config['cell_seg']['channels']:
        cell_props = apl.load_output_file(config, 'cell_props_fmt', sn, cell_chan=c_ch)
        cp_c = cell_props.shape[0]
        cell_props_area_filt = apl.load_output_file(config, 'cell_props_area_filt_fmt', sn, cell_chan=c_ch)
        cpa_c = cell_props_area_filt[cell_props_area_filt.area_thresh == 1].shape[0]
        counts += [cp_c, cpa_c]
        index += ['cell_seg','cell_area_filt']
        for s_ch in config['spot_seg']['channels']:
            spot_props = apl.load_output_file(config, 'spot_props_fmt', sn, spot_chan=s_ch)
            sp_c = spot_props.shape[0]
            # spot_props_area_filt = apl.load_output_file(config, 'spot_props_area_filt_fmt', sn, spot_chan=s_ch)
            # spa_c = spot_props_area_filt[spot_props_area_filt.area_thresh == 1].shape[0]
            spot_props_max_split = apl.load_output_file(config, 'spot_props_max_split_fmt', sn, spot_chan=s_ch)
            spm_c = spot_props_max_split.shape[0]
            spot_props_snr_filt = apl.load_output_file(config, 'spot_props_snr_filt_fmt', sn, spot_chan=s_ch, cell_chan=c_ch)
            spc_c = spot_props_snr_filt[spot_props_snr_filt.snr_thresh == 1].shape[0]
            spot_props_cid = apl.load_output_file(config, 'spot_props_cid_fmt', sn, spot_chan=s_ch, cell_chan=c_ch)
            spas_c = spot_props_cid[spot_props_cid.cell_id > 0].shape[0]
            counts += [sp_c, spm_c, spc_c, spas_c]
            _ind = ['spot_seg','spot_max_split','spot_snr_filt', 'spot_cid_filt']
            index += [i + '_ch_' + str(s_ch) for i in _ind]
    counts_table[sn] = counts
    counts_table.index = index

counts_table.to_csv(config['output_dir'] + '/counts_table.csv')
counts_table
