# Hydrogen notebook
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_09
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
project_workdir = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/mobile_elements/experiments/2022_01_22_gelgfp'
                    # Absolute path to the project work directory
image_analysis_code_path = '/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code'
seg_dir = 'segmentation/run_004/outputs'
fig_dir = 'figures'
data_dir = 'segmentation/run_004/raw_npy'
ext = '.npy'
sample_glob = '*040421*Stitch'
# %% codecell
# Imports
import sys
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
# special modules
sys.path.append(image_analysis_code_path)
import image_functions as imfn
import image_plots as ip
dir(ip)
import pipeline_segmentation.functions.spot_funcs as sf
import pipeline_segmentation.functions.segmentation_func as segf
import pipeline_segmentation.functions.image_plots as pip

# %% codecell
# =============================================================================
# Load samples
# =============================================================================
# %% codecell
# All sample names
sample_names = imfn.get_sample_names(data_dir=data_dir, sample_glob=sample_glob, ext=ext)
sample_names.sort()
sample_names
# %% codecell
# # Test sample name
# sample_names = [sample_names[24],sample_names[27]]
# sample_names
# %% codecell
# Get sample keys
factors = ['method','plasmid','fov']
keys = [imfn.get_filename_keys(sn, factors) for sn in sample_names]
keys
# %% codecell
spot_chan = 1
raws, spot_raws, cell_segs, max_props, spot_segs, cell_props = [],[],[],[],[],[]
spot_props = []
for sn in sample_names:
    # Load raw spot image
    raw = np.load(seg_dir + '/raw_npy/' + sn + '.npy')
    raws.append(raw)
    spot_raws.append(raw[:,:,spot_chan])
    # load cell seg
    cell_segs.append(np.load(seg_dir + '/cell_seg/' + sn + '_cell_seg.npy'))
    cell_props.append(pd.read_csv(seg_dir + '/cell_seg_props/' + sn + '_cell_seg_props.csv'))
    # Load max props
    max_props.append(pd.read_csv(seg_dir + '/spot_analysis/' + sn
                                 + '_max_props_cid.csv'))
     # load cell seg
    spot_segs.append(np.load(seg_dir + '/spot_seg/' + sn + '_spot_seg.npy'))

spot_props = [pd.read_csv(seg_dir + '/spot_analysis/' + sn + '/' + sn + '_cellchan_0_spotchan_1_spot_seg_cid.csv') for sn in sample_names]



# %% codecell
# =============================================================================
# Plot pixel itensity histogram
# =============================================================================
# Mask raw spot image using cell seg
spot_pixels_masked = [sr[(cs>0)] for sr, cs in zip(spot_raws, cell_segs)]
# %% codecell
# Generate a dictionairy for keys and pixels
sp_dict = imfn.get_nested_dict(keys, spot_pixels_masked, [0,1])
sp_dict
# %% codecell
# Set up ordered lists of factors
I = ['b','g']
J = ['pos', 'neg']
# Group pixel values and generate histograms
rnge = (0, 0.5)
bins = 100
hist_dict = {i:{} for i in I}
for i in I:
    for j in J:
        pix_list = sp_dict[i][j]
        pix = np.array([])
        for p in pix_list:
            pix = np.concatenate([pix,p[1]])
        y, bin_edges = np.histogram(pix, range=rnge, bins=bins)
        hist_dict[i][j] = y


# %% codecell
# plot intensities
c_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_styles=['-',':']
line_cols=[c_cycle[1], c_cycle[0]]
yticks = [2,4,6]
ylims = (-0.5,7)
xticks=[rnge[0],np.mean(rnge),rnge[1]]
ax_col='k'
# legend_labels=['Positive','Negative']
# dims=[5,2]
# lw=2
# ft=20
legend_labels=[]
lw = 1
ft = 5
dims = (0.9843 ,0.5906)
pad = 0.2
intensity_values = 0.5*(bin_edges[1:]+bin_edges[:-1]).flatten()
group_filenaming = ['HCR','splitHCR']
pixel_hist_fnt = fig_dir + '/pixel_int_hist_method_{}'
for i, gfn in zip(I, group_filenaming):
    hists = [hist_dict[i][j] for j in J]
    ip.plot_line_histogram(x_vals=intensity_values, y_vals=hists,
                             line_styles=line_styles, legend_labels=legend_labels,
                             xticks=xticks, yticks=yticks, line_cols=line_cols,
                             ax_col=ax_col, dims=dims, lw=lw, ft=ft, pad=pad)
    ip.save_png_pdf(pixel_hist_fnt.format(gfn), bbox_inches=False)
    ip.plt.show()
    ip.plt.close()


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
    for l, j in enumerate(J):
        bgms = bg_mean_dict[i][j]
        bgss = bg_mean_dict[i][j]
        shift = k*1.25 + l*0.4
        ticks.append(shift + 0.1)
        labels.append(gfn + '\n' + j)
        col = line_cols[l]
        for m, (bm, bs) in enumerate(zip(bgms, bgss)):
            t = k*2 + l - 1 + 0.1*m
            ax.errorbar(shift + 0.1*m, bm[1], yerr=bs[1], marker='.', color=col)

ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# %% codecell
# Add SNR to max props table
max_props_snr = []
for mp, bgm in zip(max_props, bg_mean):
    mp['snr'] = mp.intensity / bgm
    max_props_snr.append(mp)

# %% codecell
# =============================================================================
# Select filter for spot intensity and filter spots
# =============================================================================
# Remove maxs not associated with cells
max_dist = 5
max_props_cell = [mp[(mp.cell_dist <= max_dist)] for mp in max_props_snr]
max_props_cell[0].columns
# Generate a dictionairy for keys and maxs
mp_dict = imfn.get_nested_dict(keys, max_props_cell, [0,1])
# %% codecell
# Get threshold curves
t_lims = (0,50)  # SET
threshs = np.linspace(t_lims[0],t_lims[1],100)
curves = [[mp[mp.snr > t].shape[0] for t in threshs] for mp in max_props_cell]
curves_dict = imfn.get_nested_dict(keys, curves, [0,1])
curves_dict.keys()
[curves_dict[k].keys() for k in curves_dict.keys()]
# %% codecell
# Plot threshold curves
thresholds = (20, 10)  # SET
xlims = (0,100)
ylims = (-100,1000)
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


# %% codecell
# =============================================================================
# Check filter against roc curves
# =============================================================================
# make dict
sn_dict = imfn.get_nested_dict(keys, sample_names, [0,1])

# %% codecell
# get curves
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
        sp_df_all = pd.DataFrame([])
        cell_count = 0
        # Combine fovs
        for k, (sn, sp_tup) in enumerate(zip(sn_fovs, mpc_fovs)):
            # Get seg
            seg = np.load(seg_dir + '/cell_seg/' + sn[1] + '_cell_seg.npy')
            cell_count += np.unique(seg).shape[0]
            # Get spot params
            # sp_df_fn = seg_dir + '/spot_analysis/' + sn[1] + '_max_props_cid.csv'
            # sp_df = pd.read_csv(sp_df_fn)
            sp_df = sp_tup[1]
            sp_df['cell_id_fov'] = sp_df.cell_id.astype(str) + '_' + str(k)
            sp_df_all = sp_df_all.append(sp_df)
        # Filter by distance
        sp_df_cell = sp_df_all[(sp_df_all.cell_dist <= max_dist)]
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
# plot curves
xlims = t_lims
dims = (5,4)
roc_all_bnt = fig_dir + '/roc_curves_all_method_{}'
for i, threshold, gfn in zip(I, thresholds, group_filenaming):
    roc_df = pd.read_csv(roc_df_fnt.format(gfn))
    # plot
    fig, ax = ip.general_plot(xlabel='SNR Threshold',dims=dims)
    ax.plot(roc_df.x, roc_df.PPV,label='PPV')
    ax.plot(roc_df.x, roc_df.FOR,label='FOR')
    ax.plot(roc_df.x, roc_df.TPR, label='TPR')
    ax.plot(roc_df.x, roc_df.FPR, label='FPR')
    # ax.plot(roc_df.x, roc_df.TNR, label='TNR')
    # ax.plot(roc_df.x, roc_df.FNR,label='FNR')
    ax.plot([threshold]*2, [0,1], 'k')
    ax.legend()
    ip.save_png_pdf(roc_all_bnt.format(gfn))
    ax.set_xlim(xlims)


# %% codecell
# =============================================================================
# Plot spot histogram
# =============================================================================
# Group max values and generate histograms
rnge = t_lims
bins = 100
hist_dict = {i:{} for i in I}
for i in I:
    for j in J:
        m_list = mp_dict[i][j]
        ms = np.array([])
        for m in m_list:
            ms = np.concatenate([ms,m[1].snr.values])
        y, bin_edges = np.histogram(ms, range=rnge, bins=bins)
        hist_dict[i][j] = y

# %% codecell
# plot intensities
c_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_styles=['-',':']
legend_labels=[]
# legend_labels=['Positive','Negative']
line_cols=[c_cycle[1], c_cycle[0]]
yticks = [1,2,3]
ylims = (-0.3,4)
xticks=[rnge[0],np.mean(rnge),rnge[1]]
ax_col='k'
dims = (0.9843 ,0.5906)
# dims=[5,2]
pad=0.2
lw=1
ft=5
pixel_hist_fnt = fig_dir + '/spot_int_hist_method_{}'
intensity_values = 0.5*(bin_edges[1:]+bin_edges[:-1]).flatten()
for i, threshold, gfn in zip(I, thresholds, group_filenaming):
    hists = [hist_dict[i][j] for j in J]
    fig, ax= ip.plot_line_histogram(x_vals=intensity_values, y_vals=hists,
                             line_styles=line_styles, legend_labels=legend_labels,
                             xticks=xticks, yticks=yticks, line_cols=line_cols,
                             ax_col=ax_col, dims=dims, lw=lw, ft=ft, ylims=ylims,
                             pad=pad)
    ax.plot([threshold]*2, [0,ax.get_ylim()[1]],'-k', lw=lw)
    ip.save_png_pdf(pixel_hist_fnt.format(gfn), bbox_inches=False)
    ip.plt.show()
    ip.plt.close()


# %% codecell
# =============================================================================
# Count spots per cell and plot violins
# =============================================================================
# Filter spots by intensity
mp_dict_int_filt = {i:{j:[mp[1][mp[1].snr > thresh] for mp in mp_dict[i][j]]
                    for j in J}
                    for i, thresh in zip(I, thresholds)}
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

lw=1
ft = 5
dims = (0.9843,0.5905512)
dot_factor = 0.3

col = 'k'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
line_colors = [colors[0],colors[1]]
transparency = 0.1
plots = I
groups = ['neg', 'pos']
xticklabels = ['Control', 'Positive']
xlabel = ''
ylabel = ''
jit = 0.15
y_jit = 0.1
ylims = (-0.5,5.5)
yticks = [0,5]
pixel_hist_fnt = fig_dir + '/spot_count_violin_method_{}'
pad = 0.2
for pl, gfn in zip(plots, group_filenaming):
    fig, ax = ip.general_plot(col=col, dims=dims, lw=lw, ft=ft, pad=pad)
    xticks = ip.get_xticks(groups)
    ax.set_xticks(xticks)
    ax.set_xticklabels([],visible=False)
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_yticks(yticks)
    values = []
    for g,lc in zip(groups, line_colors):
        values.append(sc_dict[pl][g])
    ip.violin_dot_plot(ax, values, xticks, y_jit=y_jit, col=col, bw=0.5, lw=lw,
                       dot_factor=dot_factor, transparency=transparency, ft=ft,
                       colors=line_colors)
    ip.adjust_ylims(ax, lims=ylims, values=values, int_ticks=False)
    ip.save_png_pdf(pixel_hist_fnt.format(gfn), bbox_inches=False)

# %% codecell
# =============================================================================
# testing cell bbox capture
# =============================================================================

# %% codecell
# testing cell max acquisisiotn
cs = cell_segs[27]
ss = spot_segs[27]
csp = cell_props[27]
sr = spot_raws[27]
sp = spot_props[27]
row = csp.iloc[0, :]
bb = [int(i) for i in (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'])]
mask = cs[bb[0]:bb[2], bb[1]:bb[3]] > 0
r = 5
mask_conv = sf.convolve_bool_circle(mask, r=r)
plt.imshow(mask)
plt.imshow(mask_conv)

sr_pad = np.pad(sr, r).astype(np.double)
cs_pad = np.pad(cs, r)
sr_bbox = sr_pad[bb[0]: bb[2] + 2*r, bb[1]: bb[3] + 2*r]
plt.imshow(cs_pad[bb[0]: bb[2] + 2*r, bb[1]: bb[3] + 2*r])
sr_bbox.shape
mask_conv.shape
plt.imshow(sr_bbox* mask_conv)


def convolve_bool_circle(arr, r=10):
    d = 2*r + 1
    # get circle of ones
    k = np.ones((d,d))
    for i in range(d):
        for j in range(d):
            dist = ((r-i)**2 + (r-j)**2)**(1/2)
            if dist > r: k[i,j] = 0
    conv = np.pad(arr, r).astype(np.double)
    return convolve(conv, k) > 0

# %% codecell
# =============================================================================
# Use Cell spot max code to do peak local max on individual cels
# =============================================================================

def _get_merged_peaks(im, min_distance=1):
    is_peak = sf.peak_local_max(im, indices=False,
                             min_distance=min_distance) # outputs bool image
    plt.imshow(is_peak)
    plt.show()
    plt.close()
    labels = sf.label(is_peak)[0]
    merged_peaks = sf.center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
    return np.array(merged_peaks)

def get_cell_spot_maxs(cell_props, seg, spot_raw, spot_seg, r=10):
    bboxes_ids = []
    for i, row in cell_props.iterrows():
        bb = (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'], row['label'])
        bboxes_ids.append([int(b) for b in bb])
    sr_pad = np.pad(spot_raw, r,).astype(np.double)
    ss_pad = np.pad(spot_seg, r,)
    spot_max_int = []
    # Get each cell
    for bb in tqdm(bboxes_ids[1004:1005]):
        # Extract bounding box make boolean mask
        mask_ = seg[bb[0]:bb[2], bb[1]:bb[3]] > 0
        plt.imshow(mask_)
        plt.show()
        plt.close()
        mask = seg[bb[0]:bb[2], bb[1]:bb[3]] == bb[4]
        # convolve to expand mask edges
        mask_conv = convolve_bool_circle(mask, r)
        # Extract bounding box from spot image
        sr_bbox = sr_pad[bb[0]: bb[2] + 2*r, bb[1]: bb[3] + 2*r]
        ss_bbox = ss_pad[bb[0]: bb[2] + 2*r, bb[1]: bb[3] + 2*r]
        sr_bbox_masked = sr_bbox * mask_conv
        ss_bbox_masked = ss_bbox * mask_conv
        plt.imshow(sr_bbox_masked)
        plt.show()
        plt.close()
        plt.imshow(sf.label(ss_bbox_masked>0)[0], cmap='tab20', interpolation='None')
        print(np.unique(ss_bbox_masked))
        plt.show()
        plt.close()
        plm = _get_merged_peaks(sr_bbox_masked, min_distance=3)
        # plt.imshow(plm)
        # plt.show()
        # plt.close()
        # spot_max_int.append(sf.peak_local_max(sr_bbox_masked, indices=False))
        print(plm)
        # spot_max_int.append(np.max(sr_bbox_masked))
    return plm
    # return spot_max_int



spot_max_int = get_cell_spot_maxs(csp, cs, sr, ss, r=10)
# %% codecell
spot_max_int


'{a}_{b}'.format(a=5, b=6, c=9)

bool('')


from skimage.segmentation import join_segmentations

a = np.zeros((10,10))
a[2:8,2:8] = 1
a
# %% codecell
# =============================================================================
# Check how many cells have spots
# =============================================================================

cs_c = np.unique(cs).shape[0]
cs_c  # number of cells
ss_c = np.unique(ss).shape[0]
ss_c  # number of spots
sscs_c = np.unique(ss*(cs>0)).shape[0]
sscs_c  # number of spots in cells
csss_c = np.unique(cs*(ss>0)).shape[0]
csss_c  # number of cells with spots
m = sf.label(sf.peak_local_max(sr,indices=False, min_distance=3))[0]  # local maxima image
m_c = np.unique(m).shape[0]
m_c  # number of local maxima
mcs_c = np.unique(m*(cs>0)).shape[0]
mcs_c  # number of maxs in cells
cdm_c = np.unique(cs*(m>0)).shape[0]
cdm_c  # number of cells with maxs
msscs_c = np.unique(m*(ss>0)*(cs>0)).shape[0]
msscs_c  # number of maxes within spots within cells
msscs_c - sscs_c  # number of maxes in the same spot as at least one more
csss_c/cs_c  # fraction of cells with spots
cdm_c/cs_c  # Fraction of cells with maxs

# %% codecell
# =============================================================================
# FInd spots wit multiple local maxima
# =============================================================================
# Get spots in cells
sscs = ss*(cs>0)
# get maxs in spots
mss = m*(sscs>0)
# Mask spots with maxs
s_m = sscs*(m>0)
# Make 3d array with maxs and masked spots
smarr = np.dstack((s_m, mss))
# Get unique on axis 2
s_ = (0,100,0,100)
smarr_ = smarr[s_[0]: s_[1],s_[2]:s_[3],:]
m_ = mss[s_[0]: s_[1],s_[2]:s_[3]] > 0
sm = np.unique(smarr_[m_], axis=0)
sm
plt.imshow(mss[s_[0]: s_[1],s_[2]:s_[3]] > 0, interpolation='None')
plt.imshow(sscs[s_[0]: s_[1],s_[2]:s_[3]]>0, interpolation='None')

# get counts on the spot column
s_c = np.unique(sm[:,0], return_counts=True)
# get all spots with count < 1
s_c_slim = s_c[0][s_c[1] > 1]

sm_slim = pd.DataFrame([i for i in sm if i[0] in s_c_slim], columns=['spot_id','max_id'])
sm_slim

a = np.array([[[1,2,3],[3,4,3]],[[1,2,3],[3,4,3]],[[0,0,0],[0,0,0]]])

np.unique(a[np.sum(a, axis=2) > 0], axis=0)


# %% codecell
n = 3000
s_ = (0,n,0,n)

ss_ = ss[s_[0]: s_[1],s_[2]:s_[3]]
cs_ = cs[s_[0]: s_[1],s_[2]:s_[3]]
m_ = m[s_[0]: s_[1],s_[2]:s_[3]]
multimax = sf.get_spots_with_multimax(ss,cs,m)
multimax


# %% codecell
# =============================================================================
# Use Labeled max array to get max props
# =============================================================================
m = sf.label(sf.peak_local_max(sr,indices=False, min_distance=3))[0]  # local maxima image
# mp = segf.measure_regionprops(m, sr)
mp = pd.DataFrame(segf.regionprops_table(m, intensity_image = sr,
                            properties=['label','centroid']))
mp.columns
# %% codecell
sp = segf.measure_regionprops(ss, sr)


# %% codecell
# =============================================================================
# use voronoi to split multimapped spots
# =============================================================================
def generate_voronoi_diagram(width, height, centers_x, centers_y):
    # Create grid containing all pixel locations in image
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # Find squared distance of each pixel location from each center: the (i, j, k)th
    # entry in this array is the squared distance from pixel (i, j) to the kth center.
    squared_dist = (x[:, :, np.newaxis] - centers_x[np.newaxis, np.newaxis, :]) ** 2 + \
                   (y[:, :, np.newaxis] - centers_y[np.newaxis, np.newaxis, :]) ** 2
    # Find closest center to each pixel location
    return np.argmin(squared_dist, axis=2) + 1  # Array containing index of closest center

from skimage.segmentation import relabel_sequential

def split_multimax_spots(multimax, spot_props, seg, max_props, raw):
    seg_new = seg.copy()
    # subset only multimax spots
    spots = multimax.spot_id.unique()
    spot_props_multi = spot_props[spot_props.label.isin(spots)]
    # iterate through
    for index, row in spot_props_multi.iterrows():
        # print(row)
        # Extract bounding box of spot seg
        bb = (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'], row['label'])
        bb = [int(j) for j in bb]
        # mask = seg[bb[0]:bb[2], bb[1]:bb[3]] > 0
        seg_bb = seg[bb[0]:bb[2], bb[1]:bb[3]]
        # print(np.unique(seg_bb))
        mask = seg_bb == bb[4]
        # print(np.unique(mask))
        # mask = seg[bb[0]:bb[2], bb[1]:bb[3]] == bb[4]
        # Subset max props
        maxs = multimax.loc[multimax.spot_id == bb[4], 'max_id']
        max_props_sub = max_props[max_props.label.isin(maxs)]
        # print(max_props_sub)
        # Get max indices
        centers_x = max_props_sub['centroid-1'] - bb[1]
        centers_y = max_props_sub['centroid-0'] - bb[0]
        # plt.imshow(raw[bb[0]:bb[2], bb[1]:bb[3]])
        # plt.show()
        # plt.close()
        # plt.imshow(sf.label(mask)[0], interpolation='None', cmap='tab20')
        # plt.plot(centers_x, centers_y, '+w')
        # plt.show()
        # plt.close()
        # get voroni diagram
        height, width = mask.shape
        vor = generate_voronoi_diagram(width, height, centers_x, centers_y)
        # plt.imshow(vor, cmap='tab10')
        # plt.show()
        # plt.close()
        # mask voronoi
        vor_seg = vor * mask
        # plt.imshow(vor_seg, cmap='tab10')
        # plt.show()
        # plt.close()
        # get max of current spot id list
        sid = spot_props.label.max() + 1
        # assign other regions to new spot id
        for i in range(1,len(centers_x)+1):
            # print(sid)
            vor_seg[vor_seg == i] = sid
            sid += 1
        # print(np.unique(vor_seg))
        # mask out old spot in spot seg image
        seg_bbox = seg[bb[0]:bb[2], bb[1]:bb[3]]
        seg_bbox_masked = seg_bbox * (mask == 0)
        # add in new seg to the seg image
        new_seg_bbox = seg_bbox_masked + vor_seg
        seg_new[bb[0]:bb[2], bb[1]:bb[3]] = new_seg_bbox
        # plt.imshow(sf.label(seg_new[bb[0]:bb[2], bb[1]:bb[3]]>0)[0], interpolation='None', cmap='tab20')
        # print(np.unique(new_seg_bbox))
        # print(np.unique(seg_new[bb[0]:bb[2], bb[1]:bb[3]]))
    return seg_new



# %% codecell
ss = spot_segs[27]
ss_split = split_multimax_spots(multimax, sp, ss, mp, sr)

plt.imshow((sr*(ss>0))[100:300,100:300])
plt.imshow(pip.seg2rgb((ss)[100:300,100:300]))
plt.imshow(pip.seg2rgb((ss_split)[100:300,100:300]))

# %% codecell
# =============================================================================
# test full spot split
# =============================================================================

zc = (0,sr.shape[0],0,sr.shape[1])
sr_sub = sr[zc[0]:zc[1],zc[2]:zc[3]]
plt.imshow(sr_sub)
plt.show()
plt.close()
ss_sub = ss[zc[0]:zc[1],zc[2]:zc[3]]
plt.imshow(pip.seg2rgb(ss_sub))
plt.show()
plt.close()
cr_sub = raws[27][zc[0]:zc[1],zc[2]:zc[3],0]
plt.imshow(cr_sub)
plt.show()
plt.close()
cs_sub = cs[zc[0]:zc[1],zc[2]:zc[3]]
plt.imshow(pip.seg2rgb(cs_sub))
plt.show()
plt.close()
srss_sub = (sr_sub*(ss_sub>0))
min_distance = 5
m_sub_ = sf.peak_local_max(srss_sub, indices=True, min_distance=min_distance)
m_sub = sf.label(sf.peak_local_max(srss_sub, indices=False, min_distance=min_distance))[0]
plt.imshow(srss_sub)
plt.plot(m_sub_[:,1],m_sub_[:,0],'+w')
plt.show()
plt.close()
multimax_sub = sf.get_spots_with_multimax(ss_sub,cs_sub,m_sub)
mp_sub = pd.DataFrame(segf.regionprops_table(m_sub, intensity_image = sr_sub,
                            properties=['label','centroid']))
sp_sub = segf.measure_regionprops(ss_sub, sr_sub)
ss_sub_split = split_multimax_spots(multimax_sub, sp_sub, ss_sub, mp_sub, sr_sub)
plt.imshow(pip.seg2rgb(ss_sub_split))
plt.show()
plt.close()


# %% codecell



# %% codecell
# =============================================================================
# Use multimax code to assign spots to cells
# =============================================================================
def spots_to_cells(spot_seg, cell_seg):
    # Get spots in cells
    sscs = spot_seg*(cell_seg>0)
    # Mask cells with spots
    cs_masked = cs*(sscs>0)
    # Make 3d array with maxs and masked spots
    stack = np.dstack((cs_masked, sscs))
    # Get unique on axis 2
    # s_ = (0,1000,0,1000)
    # stack_ = stack[s_[0]: s_[1],s_[2]:s_[3],:]
    # sscs_ = sscs[s_[0]: s_[1],s_[2]:s_[3]] > 0
    # assgn = np.unique(stack_[sscs_ > 0], axis=0)
    assgn = np.unique(stack[sscs > 0], axis=0)
    return pd.DataFrame(assgn, columns=['cell_id','spot_id'])

# %% codecell
spot_cid_df = spots_to_cells(ss,cs)
spot_cid_df

# %% codecell
# how many spots are multimapped?
val_counts = spot_cid_df.spot_id.value_counts()
val_counts[val_counts==1].shape[0]
val_counts[val_counts==2].shape[0]
val_counts[val_counts==3].shape[0]
val_counts[val_counts==4].shape[0]

# %% codecell
# =============================================================================
# Use nearest neighbor to assign multimapped spots
# =============================================================================

def assign_multimapped_spots(spot_cid_df, spot_props, spot_raw, cell_seg):
    # Subset spot props to only thos that multimap
    val_counts = spot_cid_df.spot_id.value_counts()
    sids = val_counts[val_counts>1].index
    spot_props_sub = spot_props.loc[spot_props.label.isin(sids)]
    spot_cid_df_new = spot_cid_df.copy()
    # iterate through
    for index, row in spot_props_sub.iloc[5:6,:].iterrows():
        # get bounding box for spot
        bb = (row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3'], row['label'])
        bb = [int(j) for j in bb]
        sr_bbox = spot_raw[bb[0]:bb[2], bb[1]:bb[3]]
        plt.imshow(sr_bbox)
        plt.show()
        plt.close()
        # get spot bbox for cell seg
        cs_bbox = cell_seg[bb[0]:bb[2], bb[1]:bb[3]]
        plt.imshow(pip.seg2rgb(cs_bbox))
        plt.show()
        plt.close()
        # get a list of indices for each pixel of each cell in bbox
        cids = np.unique(cs_bbox)
        arr = np.empty((0,2))
        for c in cids:
            arr = np.concatenate([arr, np.argwhere(cs_bbox==c)], axis=0)
        y = arr[:,0]
        x = arr[:,1]
        # Find the index of the spot maximum
        m = np.argwhere(sr_bbox == np.max(sr_bbox))
        print(m)
        # Find the nearest neighbor pixel and get its cell id
        dists = (x - m[0,1]) ** 2 + (y - m[0,0]) ** 2
        ind_min = arr[np.argwhere(dists == np.min(dists)),:]
        print(ind_min)
        ind_min = np.squeeze(ind_min).astype(int)
        cid = cs_bbox[ind_min[0],ind_min[1]]
        plt.imshow(pip.seg2rgb(cs_bbox))
        plt.plot(ind_min[1], ind_min[0], '+w')
        plt.plot(m[0,1], m[0,0], 'ow')
        plt.show()
        plt.close()
        # assign that cell id to the spot props table and remove other assignments
        sid = int(row.label)
        scdfnew = pd.DataFrame({'cell_id':[cid], 'spot_id':[sid]})
        spot_cid_df_new.drop(spot_cid_df_new[spot_cid_df_new['spot_id'] == sid].index,
                                                             inplace = True)
        spot_cid_df_new = spot_cid_df_new.append(scdfnew)
    return spot_cid_df_new

# %% codecell
spot_cid_df_new = sf.assign_multimapped_spots(spot_cid_df, sp, sr, cs)



# %% codecell
# count how many cells have spots now
spot_cid_df.shape
spot_cid_df_new.shape
spot_cid_df.cell_id.unique().shape[0] / cs_c
spot_cid_df_new.spot_id.value_counts() > 1
spot_cid_df_new.spot_id.value_counts()

spot_cid_df_new[spot_cid_df_new.spot_id == 3626]
sp_m = sp.merge(spot_cid_df_new, left_on='label',
                              right_on='spot_id', how='left')

sp_m[sp_m.label==22010]







# %% codecell


sample_names[27]
