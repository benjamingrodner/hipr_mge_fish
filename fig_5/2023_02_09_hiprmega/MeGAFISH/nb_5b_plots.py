# %% md

# # Figure 3d: Image Plots

# Run after running nb_3b_filtering.py. Use the "esda" conda environment.

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
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/manuscripts/mgefish/code/fig_5/2022_12_19_hiprmegafish/MeGAFISH'

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
# ## Normalized pot count plots
# ==============================================================================

# ### Plot averaged spots per cell with the selected threshold

# %% codecell
factors = ['probe','fov']
keys = [apl.get_filename_keys(sn, factors) for sn in sample_names]
sn_dict = apl.get_nested_dict(keys, sample_names, [0])
factor_levels = ['non','orth','capsb']
sn_dict
# %% codecell
c_ch = config['cell_seg']['channels'][0]
s_ch = config['spot_seg']['channels'][0]
tab10 = apl.get_cmap_listed('tab10')
cols = [(0.5,0.5,0.5), (0,0,0), (1,0,1)]
um_per_pixel = 0.03533
dims = (1.5,0.8)
ft=6
lw=1
w=0.5
tln=0
fig_outname_fmt = config['figure_dir'] + '/{spot_chan}_spot_count_{norm}_norm_bar'
fig1, ax1 = apl.general_plot(dims=dims, ft=ft, lw=lw, tln=tln)
fig2, ax2 = apl.general_plot(dims=dims, ft=ft, lw=lw, tln=tln)
xticks = []
for i, (f, c) in enumerate(zip(factor_levels, cols)):
    fov_list = sn_dict[f]
    print(i, f)
    norm_area = []
    norm_count = []
    for fov in fov_list:
        sn = fov[1]
        print(sn)
        seg_cell = apl.load_output_file(config, 'cell_seg_area_filt_fmt', sn, cell_chan=c_ch)
        cell_props = apl.load_output_file(config, 'cell_props_area_filt_fmt', sn, cell_chan=c_ch)
        spot_props = apl.load_output_file(config, 'spot_props_cid_fmt', sn, spot_chan=s_ch, cell_chan=c_ch)
        # get area of segmented cells in microns or Get number of cells
        area_cell = seg_cell[seg_cell>0].shape[0]*(um_per_pixel**2)
        n_cells = cell_props.shape[0]
        # get number of spots with threshold
        n_spots = spot_props[spot_props.cell_id > 0].shape[0]
        # divide number of spots by cell area or number of cells
        norm_area.append(n_spots / area_cell)
        norm_count.append(n_spots / n_cells)
        # print('spots total', spot_props.shape[0])
        # print('spots with cell ids', n_spots)
        # plot values as bar plot
    bars1 = ax1.bar(i, np.mean(norm_area), color=c, width=w)
    # ax1.bar_label(bars1, labels=[str(n_cells) + '\ntotal\ncells'])
    bars2 = ax2.bar(i, np.mean(norm_count), color=c, width=w)
    xticks += [i]
# ax2.bar_label(bars2, labels=[str(n_cells) + '\ntotal\ncells'])
ax1.tick_params(axis='y', length=ft*0.5)
ax2.tick_params(axis='y', length=ft*0.5)
ax1.set_xticks(xticks)
ax2.set_xticks(xticks)
ax1.set_xticklabels([])
ax2.set_xticklabels([])
print('Cell area normalization')
plt.figure(fig1)
output_basename = fig_outname_fmt.format(spot_chan=s_ch, norm='area')
apl.save_png_pdf(output_basename)
plt.show()
print('Cell count normalization')
plt.figure(fig2)
output_basename = fig_outname_fmt.format(spot_chan=s_ch, norm='numcells')
apl.save_png_pdf(output_basename)
plt.show()



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
filt = 15
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
clims = (0,15)
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
# ## Global Moran's I statistics
# ==============================================================================

# Get spatial weights matrix

# %% codecell
# r = 20
k=20
weights_fmt = config['output_dir'] + '/' + config['weights_fmt']
for sn in sample_names:
    cell_props = apl.load_output_file(
                    config,
                    'cell_props_spot_count_outlier_filt_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    # points = seg_info.loc[:,['centroid-0','centroid-1']].values
    bool_area = (cell_props.area_thresh == 1)
    bool_sc = (cell_props.spot_count_outlier_thresh == 1)
    cell_props_ = cell_props[bool_area & bool_sc]
    points = cell_props_.centroid.apply(lambda x: eval(x))
    points = np.array([list(p) for p in points])
    w = apl.KNN(points, r)
    # w = apl.Voronoi(points)
    w_fn = weights_fmt.format(sample_name=sn, cell_chan=c_ch)
    sv = apl.save_weights(w, w_fn)
    print(w_fn)

# %% md

# Calculate global moran's I statistics and save values

# %% codecell
morans_i_fmt = config['output_dir'] + '/' + config['morans_i_fmt']
for sn in sample_names:
    w_fn = weights_fmt.format(sample_name=sn, cell_chan=c_ch)
    w = apl.read_weights(w_fn)
    cell_props = apl.load_output_file(
                    config,
                    'cell_props_spot_count_outlier_filt_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    for s_ch in config['spot_seg']['channels']:
        w.transform = 'r'
        cell_props.columns
        bool_area = (cell_props.area_thresh == 1)
        bool_sc = (cell_props.spot_count_outlier_thresh == 1)
        y = cell_props.loc[bool_area & bool_sc, 'spot_count_ch_' + str(s_ch)]
        # Moran's I calculation, save the I stat and the p value for the simulation-based null
        mi = apl.esda.moran.Moran(y,w)
        mi_ = mi.__dict__
        mi_['w'] = 'weights matrix saved as .npy file'
        mi_['y'] = mi.y.tolist()
        mi_['z'] = mi.z.tolist()
        mi_['sim'] = mi.sim.tolist()
        output_filename = morans_i_fmt.format(
                            sample_name=sn, cell_chan=c_ch, spot_chan=s_ch
                            )
        with open(output_filename, 'w') as outfile:
            apl.json.dump(mi_, outfile)
        print(output_filename)

# %% md

# Plot Moran's I simulation and statistic

# %% codecell
lw=1
ft = 6
dims = (2,1)
col = 'k'
cols = [(0.5,0.5,0.5), (0,0,0), (1,0,1)]
colors = apl.get_cmap_listed('tab10')
line_colors = [(1,0,1),(1,0,1),(0,0,0),(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0.5)]
# line_colors = [colors[0],colors[1]]
xlabel = ''
xticks = (1,1.7)
xlims = (0.65,2.05)
ylabel = ''
yticks = []
pad = 0.2
h=100

for s_ch in config['spot_seg']['channels']:
    print('Channel: ', s_ch)
    for sn, l_col in zip(sample_names, line_colors):
        fig, ax = apl.general_plot(col=col, dims=dims, lw=lw, ft=ft, pad=pad)
        print('Sample: ', sn)
        # ax.set_xticks(xticks)
        # ax.set_xticklabels([],visible=False)
        # ax.set_xlim(xlims[0],xlims[1])
        # ax.set_yticks(yticks)
        mi = apl.load_output_file(
                config, 'morans_i_fmt',sn, cell_chan=c_ch, spot_chan=s_ch
                )
        print('I: ', mi['I'])
        print('p_sim: ', mi['p_sim'])
        try:
            apl.plot_morans_i_sim(ax, mi, lw=lw, ft=ft, col=l_col, h=h)
            ax.set_xlim(-0.015, 0.015)
            ax.set_xticks([-0.01, 0, 0.01])
            ax.set_xticklabels([-0.01, 0, 0.01])
            ax.set_ylabel(None)
        except:
            print('issue plotting')
        output_fn = config['figure_dir'] + '/' + sn + '_moransI_chan_' + str(s_ch)
        apl.save_png_pdf(output_fn, bbox_inches=False)
        print(output_fn)
        plt.show()

# %% md

# ### Join count statistics

# Calculate global join counts statistics and save values

# %% codecell
join_counts_fmt = config['output_dir'] + '/' + config['join_counts_fmt']
for sn in sample_names:
    w_fn = weights_fmt.format(sample_name=sn, cell_chan=c_ch)
    w = apl.read_weights(w_fn)
    cell_props = apl.load_output_file(
                    config,
                    'cell_props_spot_count_outlier_filt_fmt',
                    sn,
                    cell_chan=c_ch
                    )
    for s_ch in config['spot_seg']['channels']:
        w.transform = 'b'
        cell_props.columns
        bool_area = (cell_props.area_thresh == 1)
        bool_sc = (cell_props.spot_count_outlier_thresh == 1)
        y = cell_props.loc[bool_area & bool_sc, 'spot_count_ch_' + str(s_ch)] > 0
        # Moran's I calculation, save the I stat and the p value for the simulation-based null
        try:
            jc = apl.esda.join_counts.Join_Counts(y,w)
        except:
            print('there was an error with the join counts calculation')
        jc_ = jc.__dict__
        for k, v in jc_.items():
            remove = ['pandas','libpysal','method']
            if any([r in str(type(v)) for r in remove]):
                jc_[k] = 'No ' + k
            elif 'numpy' in str(type(v)):
                jc_[k] = jc_[k].tolist()
        print([i + ' ' + str(type(jc_[i])) for i in jc_.keys()])
        # jc_['y'] = jc.y.tolist()
        # jc_['w'] = 'weights matrix saved as .npy file'
        # jc_['sim_bb'] = jc.sim_bb.tolist()
        # jc_['p_sim_bb'] = jc.p_sim_bb.tolist()
        # jc_['sim_bw'] = jc.sim_bw.tolist()
        # jc_['p_sim_bw'] = jc.p_sim_bw.tolist()
        # jc_['crosstab'] = 'No crosstab'
        # jc_['expected'] = 'No expected'
        output_filename = join_counts_fmt.format(
                            sample_name=sn, cell_chan=c_ch, spot_chan=s_ch
                            )
        with open(output_filename, 'w') as outfile:
            apl.json.dump(jc_, outfile)
        print(output_filename)


# %% md

# Plot Join counts simulation and statistic

# %% codecell
lw=1
ft = 6
dims = (1.1,0.5)
col = 'k'
colors = apl.get_cmap_listed('tab10')
line_colors = [(1,0,1),(1,0,1),(0.5,0.5,0.5),(0.5,0.5,0.5),(0,0,0),(0,0,0)]
xlabel = ''
xticks = (1,1.7)
xlims = (0.65,2.05)
ylabel = ''
yticks = []
pad = 0.2
h=1


join_counts_plot_fmt = config['figure_dir'] + '/{sample_name}_chan_{spot_chan}_join_counts'
for s_ch in config['spot_seg']['channels']:
    print('Channel: ', s_ch)
    for sn, l_col in zip(sample_names, line_colors):
        fig, ax = apl.general_plot(col=col, dims=dims, lw=lw, ft=ft, pad=pad)
        print('Sample: ', sn)
        # ax.set_xticks(xticks)
        # ax.set_xticklabels([],visible=False)
        # ax.set_xlim(xlims[0],xlims[1])
        # ax.set_yticks(yticks)
        jc = apl.load_output_file(
                config, 'join_counts_fmt',sn, cell_chan=c_ch, spot_chan=s_ch
                )
        print('BB: ', jc['bb'])
        print('BW: ', jc['bw'])
        print('WW: ', jc['ww'])
        print('mean_bb: ', jc['mean_bb'])
        print('p_sim_bb: ', jc['p_sim_bb'])
        try:
            apl.plot_morans_i_sim(ax, jc, sim='sim_bb', e='mean_bb', i='bb', col=l_col, h=h)
        except:
            print('there was an issue plotting simulation')
        output_fn = join_counts_plot_fmt.format(sample_name=sn, spot_chan=s_ch)
        apl.save_png_pdf(output_fn, bbox_inches=False)
        print(output_fn)
        plt.show()




















# %% md
# ==============================================================================
# ## UNUSED
# ==============================================================================
# %% md

# ### Plot average spots per cell as a function of intensity threshold

# %% codecell
t_lims = (0,150)
n = 1000
vals = 'snr'
threshs = np.linspace(t_lims[0], t_lims[1], n)
spot_snr_curve_dict = {}
for sn in sample_names:
    spot_snr_curve_dict[sn] = {}
    for s_ch in config['spot_seg']['channels']:
        cell_props = apl.load_output_file(config, 'cell_props_area_filt_fmt', sn,
                                          cell_chan=c_ch)
        spot_props = apl.load_output_file(config, 'spot_props_cid_fmt', sn,
                                            spot_chan=s_ch, cell_chan=c_ch)
        n = cell_props.shape[0]
        curve = [spot_props[spot_props[vals] > t].shape[0] / n for t in threshs]
        spot_snr_curve_dict[sn][s_ch] = curve

# %% codecell
# Plot curves
xlims=(0,50)
dims=(6,3)
lw=1
ft=6
dpi=1000
threshold_picks = [12,25,10]
for ch, thr_pk in zip(config['spot_seg']['channels'], threshold_picks):
    print('Channel: ', ch)
    for sn, c in zip(sample_names, cols):
        curve = spot_snr_curve_dict[sn][ch]
        fig, ax = apl.general_plot(dims=dims, lw=lw, ft=ft)
        ax.plot(threshs, curve, color=c)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xticks([xlims[0],(xlims[1]-xlims[0])//2, xlims[1]])
        ax.plot([thr_pk]*2, [0,1], 'k', lw=lw)
    output_basename = (fig_dir + '/chan_' + str(ch)
                        + '_spotnorm_snr_threshold_curves')
    apl.save_png_pdf(output_basename)
    plt.show()
    plt.close()


a = 1
