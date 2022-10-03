# %% md

# # Figure 3e: Image Plots

# Run after running nb_3e_filtering.py. Use the "esda" conda environment.

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
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_3/fig_3e'

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

if not os.path.exists(config['figure_dir']): os.makedirs(config['figure_dir'])

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
# ## Cell spot counts
# ==============================================================================

# Count spots in each cell and write to cell properties

# %% codecell
cell_props_spot_count_fmt = (config['output_dir'] + '/'
                            + config['cell_props_spot_count_fmt'])
for sn in sample_names:
    cell_props = apl.load_output_file(config, 'cell_props_area_filt_fmt', sn,
                                          cell_chan=c_ch)
    for s_ch in config['spot_seg']['channels']:
        spot_props = apl.load_output_file(config, 'spot_props_cid_fmt', sn,
                                            spot_chan=s_ch, cell_chan=c_ch)
        bool  = (spot_props.cell_id > 0) & (spot_props.snr_thresh > 0)
        spot_props = spot_props[bool]
        v_counts = spot_props.cell_id.value_counts()
        v_counts = v_counts.rename('spot_count_ch_' + str(s_ch)).to_frame()
        v_counts['label'] = v_counts.index
        cell_props = cell_props.merge(v_counts, on='label', how='left')
        newcol = cell_props['spot_count_ch_' + str(s_ch)].fillna(0)
        cell_props['spot_count_ch_' + str(s_ch)] = newcol
    output_filename = cell_props_spot_count_fmt.format(sample_name=sn,
                                                        cell_chan=c_ch)
    cell_props.to_csv(output_filename)
    print(output_filename)

# %% md

# Plot the distribution of spot counts in cells

# %% codecell
xlims=(0,55)
ylims=(0,10)
dims=(4,1)
fig_outname_fmt = config['figure_dir'] + '/{sample_name}_cellchan_{cell_chan}_spotchan_{spot_chan}_cell_spot_count_hist'
for sn in sample_names:
    cell_props = apl.load_output_file(config, 'cell_props_spot_count_fmt', sn,
                                          cell_chan=c_ch)
    for s_ch in config['spot_seg']['channels']:
        spot_counts = cell_props['spot_count_ch_' + str(s_ch)].values
        bins = np.max(spot_counts)
        print('Sample: ', sn)
        print('Channel: ', s_ch)
        print(bins)
        plt.figure(figsize=dims)
        plt.hist(spot_counts, bins=int(bins))
        plt.xlim(xlims[0],xlims[1])
        plt.ylim(ylims[0],ylims[1])
        fig_outname = fig_outname_fmt.format(sample_name=sn, cell_chan=c_ch, spot_chan=s_ch)
        apl.save_png_pdf(fig_outname)
        plt.show()


# %% md

# Filter cells with outlier spot counts

# %% codecell
filt = 3
cell_props_spot_count_outlier_filt_fmt = (config['output_dir'] + '/'
                            + config['cell_props_spot_count_outlier_filt_fmt'])
cell_seg_sc_outlier_filt_fmt = (config['output_dir'] + '/'
                            + config['cell_seg_sc_outlier_filt_fmt'])
for sn in sample_names:
    cell_props = apl.load_output_file(config, 'cell_props_spot_count_fmt', sn,
                                          cell_chan=c_ch)
    cell_seg = apl.load_output_file(config, 'cell_seg_area_filt_fmt', sn,
                                          cell_chan=c_ch)
    outb_list = []
    for s_ch in config['spot_seg']['channels']:
        outb_list.append(cell_props['spot_count_ch_' + str(s_ch)] < filt)
    outliers_bool = pd.Series(True, index=cell_props.index)
    for ob in outb_list:
        outliers_bool *= ob
    # Look at the cells we're removing
    cell_props_outlier = cell_props[~outliers_bool]
    # print(cell_props_outlier[['area','spot_count_ch_0','spot_count_ch_1','spot_count_ch_2']])
    # Save a new cell props table
    cell_props['spot_count_outlier_thresh'] = outliers_bool.astype(int)
    output_filename = cell_props_spot_count_outlier_filt_fmt.format(
                        sample_name=sn,
                        cell_chan=c_ch
                        )
    cell_props.to_csv(output_filename)
    print(output_filename)
    # Remove cells from seg
    seg_new = apl.filter_seg_objects(cell_seg, cell_props,
                                        'spot_count_outlier_thresh')
    output_filename = cell_seg_sc_outlier_filt_fmt.format(
                        sample_name=sn,
                        cell_chan=c_ch
                        )
    np.save(output_filename, seg_new)
    print(output_filename)
    print(np.unique(cell_seg).shape)
    print(np.unique(seg_new).shape)

# %% md

# Show the cell segmentation in terms of spot count


# %% codecell
cell_seg_spot_count_fmt = (config['output_dir'] + '/'
                            + config['cell_seg_spot_count_fmt'])
fig_outname_fmt = config['figure_dir'] + '/{sample_name}_cell_seg_spot_count'
for sn in sample_names:
    cell_seg = apl.load_output_file(
                    config,
                    'cell_seg_sc_outlier_filt_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    cell_props = apl.load_output_file(
                    config,
                    'cell_props_spot_count_outlier_filt_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    for s_ch in config['spot_seg']['channels']:
        _cp = cell_props[cell_props.spot_count_outlier_thresh == 1]
        l, sc = _cp[['label','spot_count_ch_' + str(s_ch)]].values.T
        cid_count_dict = dict(zip(l, sc))
        cell_seg_spot_count = apl.recolor_seg(cell_seg, cid_count_dict)
        # Save the segmetnation
        output_filename = cell_seg_spot_count_fmt.format(sample_name=sn, cell_chan=c_ch, spot_chan=s_ch)
        np.save(output_filename, cell_seg_spot_count)
        print(output_filename)

# %% codecell
clims = (0,3)
cmap = apl.get_cmap('autumn')(np.linspace(0,1,256))
cmap[0] = [0.5,0.5,0.5,1]
cmap = apl.ListedColormap(cmap)
cmap.set_bad('k')
fig_outname_fmt = config['figure_dir'] + '/{sample_name}_cell_seg_spot_count_spotchan_{spot_chan}'
for sn in sample_names:
    for s_ch in config['spot_seg']['channels']:
        print('Sample: ', sn)
        print('Channel: ', s_ch)
        cell_seg_spot_count = apl.load_output_file(
                                config,
                                'cell_seg_spot_count_fmt',
                                sn,
                                cell_chan=c_ch,
                                spot_chan=s_ch
                                )
        # Plot the segmentation
        fig, ax, cbar = apl.plot_image(
                        cell_seg_spot_count, cmap=cmap, discrete=True,
                        scalebar_resolution=um_per_pixel, clims=clims
                        )
        fig_outname = fig_outname_fmt.format(sample_name=sn, spot_chan=s_ch)
        apl.plt.figure(fig)
        apl.save_png_pdf(fig_outname)
        apl.plt.figure(cbar)
        apl.save_png_pdf(fig_outname + '_cbar')
        plt.show()

cell_props.shape


# %% md
# ==============================================================================
# ## Get segmentation properties for genus specific cell channel
# ==============================================================================

# new dir

# %% codecell
cell_props_genus_fmt = config['output_dir'] + '/' + config['cell_props_genus_fmt']
for sn in sample_names:
    fn = cell_props_genus_fmt.format(sample_name=sn, cell_chan='')
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


# Add genus properties to the cell props

# %% codecell
c_ch=1
g_ch=0

cell_props_genus_fmt = config['output_dir'] + '/' + config['cell_props_genus_fmt']
for sn in sample_names:
    # Load cell seg props, segmentation, raw image
    cell_props_spot_count = apl.load_output_file(
                    config,
                    'cell_props_spot_count_outlier_filt_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    cell_seg = apl.load_output_file(config, 'cell_seg_area_filt_fmt', sn,
                                          cell_chan=c_ch)
    raw = apl.load_output_file(config, 'raw_fmt',sn)
    # Generate new cell properteis based on the genus channel
    cell_props_genus_ = apl.measure_regionprops(cell_seg, raw[:,:,g_ch])
    cell_props_genus_ = cell_props_genus_[[
                            'label',
                            'max_intensity',
                            'mean_intensity',
                            'min_intensity'
                            ]]
    cell_props_genus_.columns = ['label','max_intensity_genus','mean_intensity_genus','min_intensity_genus']
    # Merge intensity properties with the old cell properties
    cell_props_genus = cell_props_spot_count.merge(
                                cell_props_genus_,
                                how='left',
                                on='label'
                                )
    output_filename = cell_props_genus_fmt.format(sample_name=sn, cell_chan=c_ch)
    cell_props_genus.to_csv(output_filename)

# %% codecell
# Plot the intensity threshold curve for genus ID and pick a threshold
t_lims = (0,0.1)
n = 100
threshs = np.linspace(t_lims[0], t_lims[1], n)
vals='mean_intensity_genus'

xlims=(0,0.1)
dims=(6,3)
lw=1
ft=6
dpi=1000
threshold_picks = [0.0025, 0.006]

for sn, thr_pk in zip(sample_names, threshold_picks):
    cell_props_genus = apl.load_output_file(
                    config,
                    'cell_props_genus_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    n = cell_props_genus.shape[0]
    curves = [[cell_props_genus[cell_props_genus[vals] > t].shape[0] / n
                                    for t in threshs]]
    print('Sample: ', sn, '\nChannel: ', c_ch)
    fig, ax = apl.plot_threshold_curves(threshs, curves, xlims=xlims,
                                        dims=dims, lw=lw, ft=ft)
    # for c in curves:
    #     slope = apl.get_curve_slope(threshs.tolist(), c)
    #     ax.plot(threshs, slope/np.max(slope))
    ax.plot([thr_pk]*2, [0,1], 'k', lw=lw)
    ax.plot(t_lims, [0,0],'k',lw=lw*0.5)
    output_basename = (config['figure_dir'] + '/' + sn + '_chan_' + str(c_ch)
                        + '_genus_int_threshold_curves')
    apl.save_png_pdf(output_basename)
    plt.show()
    plt.close()


# %% codecell
cell_props_genus_filt_fmt = config['output_dir'] + '/' + config['cell_props_genus_filt_fmt']
for sn, thr_pk in zip(sample_names, threshold_picks):
    cell_props_genus_filt = apl.load_output_file(
                    config,
                    'cell_props_genus_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    bool = cell_props_genus_filt[vals] > thr_pk
    cell_props_genus_filt['genus_thresh'] = 1*bool
    output_filename = cell_props_genus_filt_fmt.format(sample_name=sn, cell_chan=c_ch)
    cell_props_genus_filt.to_csv(output_filename, index=False)


# %% codecell
# Show raw with segmentation for filtered seg
zc = (0, 2000, 0, 2000)
clims = (0,0.25)
col=(0,1,0)
ms=5
cell_seg_genus_fmt = config['output_dir'] + '/' + config['cell_seg_genus_fmt']
for sn in sample_names:
    print(sn)
    # for ch, th_pck in zip(config['cell_seg']['channels'], threshold_picks):
    raw = apl.load_output_file(config, 'raw_fmt', sn, c_ch)[:,:,c_ch]
    seg = apl.load_output_file(config, 'cell_seg_area_filt_fmt', sn, c_ch)
    seg_new = seg.copy()
    props = apl.load_output_file(
                                config,
                                'cell_props_genus_filt_fmt',
                                sn,
                                cell_chan=c_ch
                                )
    remove_cids = props.loc[props.genus_thresh == 0, ['label','bbox']].values
    for i, (c, b) in enumerate(remove_cids):
        b = eval(b)
        b_sub = seg_new[b[0]:b[2],b[1]:b[3]]
        b_sub = b_sub * (b_sub != c)
        seg_new[b[0]:b[2],b[1]:b[3]] = b_sub
    output_filename = cell_seg_genus_fmt.format(sample_name=sn,
                                                    cell_chan=c_ch)
    np.save(output_filename, seg_new)
    fig, ax, cbar = apl.plot_image(raw, zoom_coords=zc, cmap='inferno', clims=clims)
    apl.plot_seg_outline(ax, seg_new, col=col)
    sc = apl.load_output_file(
            config,
            'spot_props_cid_fmt',
            sn,
            spot_chan=s_ch,
            cell_chan=c_ch
            )
    props_ = props[['label','genus_thresh']]
    sc = sc.merge(props_, how='left',left_on='cell_id',right_on='label')
    sc = sc.centroid.apply(lambda x: eval(x))
    # sc = sc[sc.genus_thresh > 0].centroid.apply(lambda x: eval(x))
    sc = np.array([[i for i in k] for k in sc])
    ax.plot(sc[:,1], sc[:,0], '.', ms=ms, color=(1,1,1))
    plt.show()
    plt.close()


# %% md

# Get a dictionary of sample names and pick the order of the important factors

# %% codecell
factors = ['probes','fov']
keys = [apl.get_filename_keys(sn, factors) for sn in sample_names]
sn_dict = apl.get_nested_dict(keys, sample_names, [0])
factor_levels = ['strep','laut']
sn_dict

# %% codecell

# Plot bar
# %% codecell
tab10 = apl.get_cmap_listed('tab10')
cols = [(0.5,0.5,0.5), (1,1,0)]
um_per_pixel = 0.03533
dims = (0.9,1)
ft=6
lw=1
w=0.2
tln=0
fig_outname_fmt = config['figure_dir'] + '/genus_mge_fraction'
fig1, ax1 = apl.general_plot(dims=dims, ft=ft, lw=lw, tln=tln)
xticks = []
jit = 0.2
maxs = []
for i, f in enumerate(factor_levels):
    fov_list = sn_dict[f]
    print(i, f)
    frac_neg = []
    frac_pos = []
    for fov in fov_list:
        sn = fov[1]
        print(sn)
        cell_props = apl.load_output_file(
                                    config,
                                    'cell_props_genus_filt_fmt',
                                    sn,
                                    cell_chan=c_ch
                                    )
        # get area of segmented cells in microns or Get number of cells
        n_gneg = cell_props[cell_props['genus_thresh'] == 0].shape[0]
        n_gpos = cell_props[cell_props['genus_thresh'] == 1].shape[0]
        booln = (cell_props['genus_thresh'] == 0) & (cell_props['spot_count_ch_' + str(s_ch)] > 0)
        n_gneg_spos = cell_props[booln].shape[0]
        boolp = (cell_props['genus_thresh'] == 1) & (cell_props['spot_count_ch_' + str(s_ch)] > 0)
        n_gpos_spos = cell_props[boolp].shape[0]
        # get number of spots with threshold
        # divide number of spots by cell area or number of cells
        frac_neg.append(n_gneg_spos / n_gneg)
        frac_pos.append(n_gpos_spos / n_gpos)
        # print('spots total', spot_props.shape[0])
        # print('spots with cell ids', n_spots)
        # plot values as bar plot

    bars1 = ax1.bar(i-jit, np.mean(frac_neg), color=cols[0], width=w)
    bars1 = ax1.bar(i+jit, np.mean(frac_pos), color=cols[1], width=w, edgecolor=(0,0,0))
    maxs.append(np.max([np.mean(frac_neg),np.mean(frac_neg)]))
    xticks += [i]
ax1.set_yticks([0,0.1,0.2,0.3])
# ax2.bar_label(bars2, labels=[str(n_cells) + '\ntotal\ncells'])
ax1.tick_params(axis='y', length=ft*0.5)
ax1.set_xticks(xticks)
ax1.set_xticklabels([])
print('Cell area normalization')
plt.figure(fig1)
output_basename = fig_outname_fmt
apl.save_png_pdf(output_basename)
plt.show()




# %% md
# ==============================================================================
# ## Generate fraction barplots
# ==============================================================================

# Get a dictionary of sample names and pick the order of the important factorsy

# %% codecell
factors = ['probes','fov']
keys = [apl.get_filename_keys(sn, factors) for sn in sample_names]
sn_dict = apl.get_nested_dict(keys, sample_names, [0])
factor_levels = ['strep','laut']
sn_dict

# %% md
