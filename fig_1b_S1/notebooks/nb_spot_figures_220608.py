# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_06_08
# =============================================================================
"""
The notebook Has Been Built for...generating figures for spot segmentation of
the GFP E. coli experiment

For use with the hiprfish_imaging_py38 conda env
"""
# %% codecell
# =============================================================================
# Setup
# =============================================================================
# Modify
project_workdir = '/workdir/bmg224/manuscripts/mgefish/code/fig_1/runs/run_004'
                    # Absolute path to the project work directory

# %% codecell
# Imports
import sys
import os
import gc
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


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
config_fn = 'config.yaml'
with open(config_fn, 'r') as f:
    config = yaml.safe_load(f)

image_analysis_code_path = config['pipeline_path']
seg_dir = config['output_dir']
fig_dir = seg_dir + '/../figures'
data_dir = config['input_dir']
ext = config['input_ext']

# %% codecell
# special modules
sys.path.append(image_analysis_code_path + '/' + config['functions_path'])
import image_functions as imfn
import image_plots as ip
import spot_funcs as sf
import segmentation_func as segf
import image_plots as pip

# %% codecell
# =============================================================================
# combine sample names
# =============================================================================
def get_sns(sample_glob, methods, factor='method'):
    sample_names_ = imfn.get_sample_names(data_dir=data_dir, sample_glob=sample_glob, ext=ext)
    methods = [factor + '_' + i for i in methods]
    return [sn for sn in sample_names_ if any(x in sn for x in methods)]

sample_names = []

sample_glob = '*040421*Stitch'
methods = ['b','g']
sample_names += get_sns(sample_glob, methods)
sample_names.sort()

# %% codecell
sample_glob = '*080220*Stitch'
methods = ['a','b','c1']
sample_names += get_sns(sample_glob, methods)
sample_names.sort()

# %% codecell
sample_glob = '*pk*stitch'
methods = ['pos']
sample_names += get_sns(sample_glob, methods, factor='pk')
sample_names.sort()
sample_names

# %% codecell
# Get keys
factors = ['gelgfp','exp','method', 'plasmid','pk','gfp','fov']
keys = [imfn.get_filename_keys(sn, factors) for sn in sample_names]
keys = [[i[0] + i[1], i[2], i[3]] for i in keys]
keys

# %% codecell
# get sort order
I = ['080220a','080220b', '080220c1','040421b','040421g','pkpos']
group_filenaming = ['Single','Ten','Branched','HCR','HCRsplit','Gel']
J = ['neg','pos']
line_cols = list(plt.get_cmap('tab10').colors[:2])
# line_cols.reverse()

# %% codecell
# Get files
raws = [np.load(seg_dir + '/raw_npy/' + sn + '.npy') for sn in sample_names]

spot_raws = [r[:,:,1] for r in raws]
spot_segs = [np.load(seg_dir + '/spot_seg/' + sn + '/' + sn + '_chan_1_spot_seg.npy') for sn in sample_names]
spot_props = [pd.read_csv(seg_dir + '/spot_analysis/' + sn + '/' + sn + '_cellchan_0_spotchan_1_spot_seg_props_cid.csv') for sn in sample_names]

cell_raws = [r[:,:,0] for r in raws]
cell_segs = [np.load(seg_dir + '/cell_seg/' + sn + '/' + sn + '_chan_0_cell_seg.npy') for sn in sample_names]
cell_props = [pd.read_csv(seg_dir + '/cell_seg_props' + '/' + sn + '/' + sn + '_chan_0_cell_seg_props.csv') for sn in sample_names]

# %% codecell
# =============================================================================
# Calculate SNR for spots
# =============================================================================
# Get background values
bg = [r[s == 0] for s, r in zip(spot_segs, spot_raws)]
bg_mean = [np.mean(i) for i in bg]
bg_std = [np.std(i) for i in bg]

# %% codecell
# PLot background values
bg_mean_dict = imfn.get_nested_dict(keys, bg_mean, [0,1])
bg_std_dict = imfn.get_nested_dict(keys, bg_std, [0,1])
fig, ax = ip.general_plot()
ticks = []
labels = []
for k, (i, gfn) in enumerate(zip(I, group_filenaming)):
    shift = k*1.25
    ticks.append(shift + 0.3)
    labels.append(gfn)
    for l, j in enumerate(J):
        bgms = bg_mean_dict[i][j]
        bgss = bg_mean_dict[i][j]
        shift2 = shift + l*0.4
        col = line_cols[l]
        for m, (bm, bs) in enumerate(zip(bgms, bgss)):
            ax.errorbar(shift2 + 0.1*m, bm[1], yerr=bs[1], marker='.', color=col)

ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# %% codecell
# Add SNR to  props table
spot_props_snr = []
for mp, bgm in zip(spot_props, bg_mean):
    mp['snr'] = mp.max_intensity / bgm
    spot_props_snr.append(mp)

# %% codecell
# =============================================================================
# Select filter for spot intensity and filter spots
# =============================================================================
# Remove maxs not associated with cells
spot_props_snr[0].shape
spot_props_cell = [mp[~mp.cell_id.isnull()] for mp in spot_props_snr]
spot_props_cell[0].shape
# Generate a dictionairy for keys and maxs
mp_dict = imfn.get_nested_dict(keys, spot_props_cell, [0,1])
np.max([np.max(s.snr.values) for s in spot_props_snr])
# %% codecell
# Get threshold curves
t_lims = (0,50)  # SET
threshs = np.linspace(t_lims[0],t_lims[1],100)
curves = [[mp[mp.snr > t].shape[0] for t in threshs] for mp in spot_props_cell]
curves_dict = imfn.get_nested_dict(keys, curves, [0,1])
curves_dict.keys()
[curves_dict[k].keys() for k in curves_dict.keys()]

# %% codecell
# Plot threshold curves
thresholds = (5, 4, 4.5, 8, 3, 4)  # SET
xlims = (0,10)
ylims = (-100,10000)
for i, threshold in zip(I, thresholds):
    fig, ax = ip.general_plot(col='k', dims=(10,3))
    for j in J:
        curve_list = curves_dict[i][j]
        for k, c in curve_list:
            ax.plot(threshs, c, label=", ".join(k), lw=3)
            ax.legend(fontsize=10)
        ax.plot([threshold]*2, [0,100000],'-k')
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ip.plt.show()
    ip.plt.close()

# Plot ROC curves

# Plot histograms

# count spots per cell and plot violins


# %% codecell
# =============================================================================
# Check filter against roc curves
# =============================================================================
# make dict
sn_dict = imfn.get_nested_dict(keys, sample_names, [0,1])
cs_dict = imfn.get_nested_dict(keys, cell_segs, [0,1])
# %% codecell
# get curves
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
x = threshs  # thresholds
roc_df_fnt = fig_dir + '/ROC_df_method_{}.csv'
# iterate through methods
for i, gfn in zip(I, group_filenaming):
    rd = {}
    # iterate through pos/neg
    for j in J:
        print(i,j)
        sn_fovs = sn_dict[i][j]
        mpc_fovs = mp_dict[i][j]
        cs_fovs = cs_dict[i][j]
        sp_df_all = pd.DataFrame([])
        cell_count = 0
        # Combine fovs
        for k, (sn, sp_tup, cs_tup) in enumerate(zip(sn_fovs, mpc_fovs, cs_fovs)):
            # Get seg
            seg = cs_tup[1]
            cell_count += np.unique(seg).shape[0]
            # Get spot params
            # sp_df_fn = seg_dir + '/spot_analysis/' + sn[1] + '_max_props_cid.csv'
            # sp_df = pd.read_csv(sp_df_fn)
            sp_df = sp_tup[1]
            sp_df['cell_id_fov'] = sp_df.cell_id.astype(str) + '_' + str(k)
            sp_df_all = sp_df_all.append(sp_df)
        # Filter by distance
        sp_df_cell = sp_df_all[~sp_df_all.cell_id.isnull()]
        # Get threshold curves
        # nsrs = [sp_df_cell.loc[(sp_df_cell.intensity < l),'cell_id_fov'] for l in x]
        psrs = [sp_df_cell.loc[(sp_df_cell.snr >= l),'cell_id_fov'] for l in x]
        rd[j] = {'c':cell_count, 'p':psrs}
        # rd[j] = {'c':cell_count, 'n':nsrs,'p':psrs}
    # calculate values
    # Rs = {}
    # for j in J:
    FPR = [ns.unique().shape[0] / rd['neg']['c'] for ns in rd['neg']['p']]
    TNR = [1-fpr for fpr in FPR]
    TPR = [ns.unique().shape[0] / rd['pos']['c'] for ns in rd['pos']['p']]
    FNR = [1 - tpr for tpr in TPR]
        # Rs[j + '_NR'] = NR
        # Rs[j + '_PR'] = PR
    PPV = [ps.unique().shape[0] / (ps.unique().shape[0] + nps.unique().shape[0] + 1e-15)
            for ps, nps in zip(rd['pos']['p'], rd['neg']['p'])]
    FOR = [(rd['pos']['c'] - ps.unique().shape[0]) / ((rd['pos']['c'] -\
            ps.unique().shape[0]) + (rd['neg']['c'] - nps.unique().shape[0]) + 1e-15)
            for ps, nps in zip(rd['pos']['p'], rd['neg']['p'])]
    # Save values
    roc_df = pd.DataFrame({'x':x,'TNR':TNR,'FPR':FPR,'FNR':FNR,'TPR':TPR,'PPV':PPV,'FOR':FOR})
    roc_df.to_csv(roc_df_fnt.format(gfn), index=False)

# %% codecell
# Pick filter based on FPR TPR difference
ind_special = [0,1]
val_special = [5,5]
thr_auto = []
for j, (i, threshold, gfn) in enumerate(zip(I, thresholds, group_filenaming)):
    roc_df = pd.read_csv(roc_df_fnt.format(gfn))
    if j not in ind_special:
        diff = roc_df.TPR.values - roc_df.FPR.values
        index = np.max(np.argwhere(diff == np.max(diff)))
    else:
        i_s = np.argwhere(np.array(ind_special) == j)[0][0]
        index = np.argmin(np.abs(roc_df.x.values - val_special[i_s]))
    thr_auto.append(threshs[index])
    print(i, gfn, threshs[index])
    print('TPR: ', roc_df.TPR.values[index])
    print('FPR: ', roc_df.FPR.values[index])





# %% codecell
# plot curves
xlims = (0,20)
ylims = (0,1)
dims = (0.827 ,0.511)
ft = 6
lw = 1
c_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
thr_manual = [t for t in thr_auto[:4]] + [5,5]
# thr = thr_auto
thr = thr_manual
# dims = (5,4)
roc_all_bnt = fig_dir + '/roc_curves_all_method_{}'
for j, (i, threshold, gfn) in enumerate(zip(I, thr, group_filenaming)):
    roc_df = pd.read_csv(roc_df_fnt.format(gfn))
    # plot
    fig, ax = ip.general_plot(dims=dims, lw=lw, pad=0, ft=ft)
    # ax.plot(roc_df.x, roc_df.PPV,label='PPV')
    # ax.plot(roc_df.x, roc_df.FOR,label='FOR')
    ax.plot(roc_df.x, roc_df.TPR, label='TPR', color=c_cycle[1])
    ax.plot(roc_df.x, roc_df.FPR, label='FPR', color=c_cycle[0], linestyle=':')
    # ax.plot(roc_df.x, roc_df.TNR, label='TNR')
    # ax.plot(roc_df.x, roc_df.FNR,label='FNR')
    if threshold:
        idx = (np.abs(threshs - threshold)).argmin()
        threshold = threshs[idx]
        print('TPR: ', roc_df.TPR.values[idx])
        print('FPR: ', roc_df.FPR.values[idx])
        ax.plot([threshold]*2, [0,1], 'k', lw=lw)
    # if j == 0: ax.legend(fontsize=ft)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(ticks=[0,10,20], labels=[])
    ax.set_yticks(ticks=[0,0.5,1], labels=[])
    ax.set_xticks([xlims[0],(xlims[1]-xlims[0])//2, xlims[1]])
    ip.save_png_pdf(roc_all_bnt.format(gfn))
    plt.show()
    plt.close()

# %% codecell
# Plot ROC
dims = (4,4)
roc_all_bnt = fig_dir + '/roc_curves'
fig, ax = ip.general_plot(dims=dims)
for i, threshold, gfn in zip(I, thresholds, group_filenaming):
    roc_df = pd.read_csv(roc_df_fnt.format(gfn))
    ax.plot(roc_df.FPR, roc_df.TPR,label=gfn)
ax.legend()
ip.save_png_pdf(roc_all_bnt.format(gfn))


# %% codecell
# =============================================================================
# Count spots per cell and plot violins
# =============================================================================
# Filter spots by intensity
mp_dict_int_filt = {i:{j:[mp[1][mp[1].snr > thresh] for mp in mp_dict[i][j]]
                    for j in J}
                    for i, thresh in zip(I, thr_manual)}
# Count spots per cell
cell_ids = [np.unique(cs) for cs in cell_segs]
cid_dict = imfn.get_nested_dict(keys, cell_ids, [0,1])
sc_dict = {}
for i in I:
    sc_dict[i] = {}
    for j in J:
        sc_dict[i][j] = []
        mps = mp_dict_int_filt[i][j]
        cids_l = cid_dict[i][j]
        spot_count = []
        for mp, cids in zip(mps, cids_l):
            for cid in cids[1]:
                spot_count.append(mp[mp.cell_id == cid].shape[0])
        sc_dict[i][j] += spot_count
# %% codecell
# Plot violins
# lw=2
# ft = 12
# dims = (3,2)
# dot_factor = 1

a = mp_dict['pkpos']['pos'][0][1]
b = a[(a.snr > 3)]
c = a[~a.cell_id.isnull()]
a.shape
b.shape
a.cell_id.unique().shape
b.cell_id.unique().shape
np.unique(cs_dict['pkpos']['pos'][0][1]).shape

np.mean(sc_dict['pkpos']['neg'])


lw=1
ft = 6
# dims = (0.9843,0.55)
dims = (0.875 ,0.565)
dot_factor = 0.3
transparency = 0.05
scat = False
alpha=1
col = 'k'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_colors = [colors[0],colors[1]]

plots = I
groups = ['neg', 'pos']
xticklabels = ['Control', 'Positive']
xlabel = ''
xsp = 0.1
# xticks = ip.get_xticks(groups, spacer=xsp)
xticks = (1,1.7)
xlims = (0.65,2.05)
ylabel = ''
jit = 0.5
y_jit = 0.05
ylims = (-0.5,5.5)
yticks = np.arange(6)
pixel_hist_fnt = fig_dir + '/spot_count_violin_method_{}'
pad = 0.2
for pl, gfn in zip(plots, group_filenaming):
    fig, ax = ip.general_plot(col=col, dims=dims, lw=lw, ft=ft, pad=pad)
    ax.set_xticks(ticks=xticks, labels=[])
    # ax.set_xticklabels([],visible=False)
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_yticks(ticks=yticks, labels=[])
    # ax.set_yticks(ticks=yticks, family='Arial')
    values = []
    for g,lc in zip(groups, line_colors):
        values.append(sc_dict[pl][g])
    ip.violin_dot_plot(ax, values, xticks, y_jit=y_jit, col=col, bw=0.5, lw=lw,
                       dot_factor=dot_factor, transparency=transparency, ft=ft,
                       colors=line_colors, scat=scat, alpha=alpha)
    ip.adjust_ylims(ax, lims=ylims, values=values, int_ticks=False)
    ip.save_png_pdf(pixel_hist_fnt.format(gfn), bbox_inches=False)
    # ip.save_png_pdf(pixel_hist_fnt.format(gfn))


# %% codecell
# Plot histograms
# lw=2
# ft = 12
# dims = (3,2)
# dot_factor = 1

a = mp_dict['pkpos']['pos'][0][1]
b = a[(a.snr > 3)]
c = a[~a.cell_id.isnull()]
a.shape
b.shape
a.cell_id.unique().shape
b.cell_id.unique().shape
np.unique(cs_dict['pkpos']['pos'][0][1]).shape

np.mean(sc_dict['pkpos']['neg'])


lw=1
ft = 6
# dims = (0.9843,0.55)
dims = (0.875 ,0.565)
# dims = (5 ,3)
dot_factor = 0.3
transparency = 0.05
scat = False
alpha=1
col = 'k'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_colors = [colors[0],colors[1]]

plots = I
groups = ['neg', 'pos']
xticklabels = ['Control', 'Positive']
xlabel = ''
ylims = (-0.5,5.5)
yticks = np.arange(6)
pixel_hist_disc_fnt = fig_dir + '/spot_count_hist_method_{}'
pad = 0.2
bar_height = 0.95
mean_lw = lw
mean_adj = 0.75
for pl, gfn in zip(plots, group_filenaming):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True,
            figsize=(dims[0], dims[1]), tight_layout={'pad': pad})
    # ax.set_xticks(ticks=xticks, labels=[])
    # ax.set_xticklabels([],visible=False)
    # ax.set_ylim(ylims[0],ylims[1])
    # ax.set_xlim(xlims[0],xlims[1])
    # ax.set_yticks(ticks=yticks, labels=[])
    # ax.set_yticks(ticks=yticks, family='Arial')
    for g,lc, ax in zip(groups, line_colors, axs):
        values = sc_dict[pl][g]
        print(gfn, g)
        print('n_cells = ', len(values))
        m = np.max(values)
        bins = np.arange(m) if m>1 else np.arange(2)
        hist = np.histogram(values, bins=bins)
        ax.barh(hist[1][:-1], hist[0]/2, height=bar_height, color=lc)
        ax.barh(hist[1][:-1], -hist[0]/2, height=bar_height, color=lc)
        ax.spines['top'].set_color(None)
        ax.spines['bottom'].set_color(col)
        ax.spines['right'].set_color(None)
        if g=='neg':
            ax.spines['left'].set_color(col)
            ax.tick_params(axis='y', direction='in', length=ft/3, labelsize=ft,
                    color=col, labelcolor=col)
            ax.set_yticks(ticks=yticks, labels=[])
        else:
            ax.spines['left'].set_color(None)
            ax.tick_params(axis='y', length=0)
        ymean = np.mean(values)
        m_hist = np.max(hist[0])
        ax.hlines(ymean, -m_hist/2*mean_adj, m_hist/2*mean_adj, colors='red', linewidth=mean_lw)
        ax.tick_params(axis='x', length=0, labelbottom=None)
        ax.set_ylim(ylims[0],ylims[1])


    # ip.save_png_pdf(pixel_hist_disc_fnt.format(gfn), bbox_inches=False)























# %% codecell
# =============================================================================
# Check regiionprops measurment
# =============================================================================

from skimage.measure import regionprops
sp_ = regionprops(ss, intensity_image = sr)
properties=['label','centroid','area','max_intensity','mean_intensity','min_intensity', 'bbox','major_axis_length', 'minor_axis_length','orientation','eccentricity','perimeter']
df = pd.DataFrame([])
for p in properties:
    df[p] = [s[p] for s in sp_]
for j in range(2):
    df['centroid-' + str(j)] = [r['centroid'][j] for i, r in df.iterrows()]
for j in range(4):
    df['bbox-' + str(j)] = [r['bbox'][j] for i, r in df.iterrows()]
df['bbox-1']




# %% codecell
