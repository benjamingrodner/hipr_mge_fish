import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.cm import get_cmap
import networkx as nx
import json
import libpysal
# from libpysal.cg.voronoi  import voronoi, voronoi_frames
import numpy as np
import os
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from copy import copy
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colors
from libpysal.weights.contiguity import Voronoi
from libpysal.weights import Kernel, DistanceBand, KNN
import esda
import re
from collections import defaultdict
from scipy.stats import gaussian_kde


# Take pysal W object and write to json file
def save_weights(w, fn):
    # convert to nx graph
    w_nx = w.to_networkx()
    # convert to json serializable format
    w_data = nx.readwrite.json_graph.node_link_data(w_nx)
    # Replace int32 values with int
    for i in range(len(w_data['links'])):
        w_data['links'][i]['target'] = int(w_data['links'][i]['target'])
    # Write to json
    file = open(fn, 'w+')
    s1 = json.dump(w_data, file)
    print('Wrote: ', fn)
    return

# Read json node-link format weights file and convert to pysal W object
def read_weights(fn):
    file_ = open(fn)
    w_data_ = json.load(file_)
    w_nx_ = nx.readwrite.json_graph.node_link_graph(w_data_)
    w_ = libpysal.weights.W.from_networkx(w_nx_)
    return w_


def get_cmap_listed(cmap):
    return get_cmap(cmap).colors

def general_plot(xlabel='', ylabel='', ft=12, dims=(5,3), col='k', lw=1, pad=0, tln=0.25):
    fig, ax = plt.subplots(figsize=(dims[0], dims[1]),  tight_layout={'pad': pad})
    for i in ax.spines:
        ax.spines[i].set_linewidth(lw)
    ax.spines['top'].set_color(col)
    ax.spines['bottom'].set_color(col)
    ax.spines['left'].set_color(col)
    ax.spines['right'].set_color(col)
    ax.tick_params(direction='in', labelsize=ft, color=col, labelcolor=col, length=ft*tln)
    ax.set_xlabel(xlabel, fontsize=ft, color=col)
    ax.set_ylabel(ylabel, fontsize=ft, color=col)
    ax.patch.set_alpha(0)
    return(fig, ax)


def single_color_cmap(col):
    return LinearSegmentedColormap.from_list('temp', [col], 1)


def plot_morans_i_sim(ax, mi, lw=1, col='k',ft=12, h=1, sim='sim', e='EI',i='I'):
    # cmap = single_color_cmap(sim_col)
    # pd.DataFrame(mi[sim], columns=['Simulation']).plot.kde(
    #                                                 ax=ax,
    #                                                 color=col,
    #                                                 legend=False
    #                                                 )
    vals = mi[sim]
    kernel = gaussian_kde(vals)
    rng = np.max(np.abs([np.min(vals), np.max(vals)]))
    x = np.linspace(mi[e]-rng, mi[e]+rng, 1000)
    pdf = kernel.evaluate(x)
    ax.plot(x,pdf, color=col, lw=lw)
    ax.fill_between(x,0,pdf, facecolor=col,alpha=0.5)
    # pd.DataFrame(mi['sim'], columns=['Simulation']).plot.kde(ax=ax, cmap=cmap)
    # l = np.unique(mi['sim'])
    ylims = ax.get_ylim()
    ax.plot([mi[e]]*2, [0,ylims[1]*3/5], label='Expected', color='k', lw=lw*0.75)
    ax.plot([mi[i]]*2, [0,ylims[1]/2], label='Observed', color=col, lw=lw*0.75)
    # ax.legend()
    return ax


def save_png_pdf(basename, bbox_inches='tight', dpi=1000):
    for ext in ['.pdf','.png']:
        fn = basename + ext
        if bbox_inches:
            plt.savefig(fn, transparent=True, bbox_inches=bbox_inches, dpi=dpi)
        else:
            plt.savefig(fn, transparent=True, dpi=dpi)


def plot_threshold_curves(threshs, curves, xlims, dims=(5,2), lw=3,
                          ft=12):
    fig, ax = general_plot(col='k', dims=dims)
    ax.plot(xlims, [1,1],'k',lw=lw*0.75)
    for c in curves:
        ax.plot(threshs, c, lw=lw)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_xticks([xlims[0],(xlims[1]-xlims[0])//2, xlims[1]])
    # ax.set_xticklabels([])
    return fig, ax


def get_curve_dict(sample_names, config, seg_type, fmt, vals, threshs, lessthan=False):
    curve_dict = {}
    for sn in sample_names:
        curve_dict[sn] = {}
        for ch in config[seg_type]['channels']:
            props_fn = (config['output_dir'] + '/'
                        + config[fmt].format(sample_name=sn, spot_chan=ch, cell_chan=ch))
            props = pd.read_csv(props_fn)
            n = props.shape[0]
            if lessthan:
                curve = [props[props[vals] < t].shape[0] / n for t in threshs]
            else:
                curve = [props[props[vals] > t].shape[0] / n for t in threshs]
            curve_dict[sn][ch] = curve
    return curve_dict


def load_output_file(config, fmt, sample_name='', cell_chan='', spot_chan=''):
    fn = (config['output_dir'] + '/'
                + config[fmt].format(sample_name=sample_name,
                                         cell_chan=cell_chan,
                                         spot_chan=spot_chan))
    ext = os.path.splitext(fn)[1]
    if ext == '.npy':
        return np.load(fn)
    elif ext == '.csv':
        return pd.read_csv(fn)
    elif ext == '.json':
        with open(fn) as f:
            file = json.load(f)
        return file
    else:
        raise ValueError('must be csv or npy file')
    return


def get_curve_slope(x, y):
    y_, x_ = [i[1:] + [i[-1]] for i in [y,x]]
    y, x, y_, x_ = [np.array(i) for i in [y, x, y_, x_]]
    return (y_ - y) / (x_ - x + 1e-5)


def _image_figure(dims, dpi=500):
    fig = plt.figure(figsize=(dims[0], dims[1]))
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.set_axis_off()
    fig.add_axes(ax)
    return(fig, ax)


def plot_image(
            im, im_inches=5, cmap=(), clims=('min','max'), zoom_coords=(), scalebar_resolution=0,
            axes_off=True, discrete=False, cbar_ori='horizontal', dpi=500
        ):
    s = im.shape
    dims = (im_inches*s[1]/np.max(s), im_inches*s[0]/np.max(s))
    fig, ax = _image_figure(dims, dpi=dpi)
    im_ = im[~np.isnan(im)]
    llim = np.min(im_) if clims[0]=='min' else clims[0]
    ulim = np.max(im_) if clims[1]=='max' else clims[1]
    clims = (llim, ulim)
    if cmap:
        ax.imshow(im, cmap=cmap, clim=clims, interpolation="none")
    else:
        ax.imshow(im, interpolation="none")
    zc = zoom_coords if zoom_coords else (0,im.shape[0],0,im.shape[1])
    ax.set_ylim(zc[1],zc[0])
    ax.set_xlim(zc[2],zc[3])
    if axes_off:
        ax.set_axis_off()
    if scalebar_resolution:
        scalebar = ScaleBar(
                scalebar_resolution, 'um', frameon = False,
                color = 'white', box_color = 'white'
            )
        plt.gca().add_artist(scalebar)
    cbar = []
    fig2 = []
    if cmap:
        if cbar_ori == 'horizontal':
            fig2 = plt.figure(figsize=(dims[0], dims[0]/10))
        elif cbar_ori == 'vertical':
            fig2 = plt.figure(figsize=(dims[1]/10, dims[1]))
        if discrete:
            vals = np.sort(np.unique(im))
            vals = vals[~np.isnan(vals)]
            vals = vals[(vals>=clims[0]) & (vals<=clims[1])]
            cbar = get_discrete_colorbar(vals, cmap, clims)
        else:
            image=plt.imshow(im, cmap=cmap, clim=clims)
            plt.gca().set_visible(False)
            cbar = plt.colorbar(image,orientation=cbar_ori)
    return(fig, ax, fig2)


def get_discrete_colorbar(vals, cmp, clims, integers=True):
    l = clims[1] - clims[0]
    # l = max(vals)-min(vals)
    cmp_ = get_cmap(cmp,lut=int(l+1))
    cmp_bounds = np.arange(int(l+2)) - 0.5
    norm = colors.BoundaryNorm(cmp_bounds,cmp_.N)
    image=plt.imshow(np.array([list(vals)]), cmap=cmp_, clim=clims, norm=norm)
    plt.gca().set_visible(False)
    cbar = plt.colorbar(image,ticks=vals,orientation="horizontal")
    if integers:
        cbar.set_ticklabels([str(int(v)) for v in vals])
    else:
        cbar.set_ticklabels([str(v) for v in vals])
    return(cbar)


def seg2rgb(seg):
    return label2rgb(seg,  bg_label = 0, bg_color = (0,0,0))

def plot_seg_outline(ax, seg, col=(0.5,0.5,0.5)):
    cmap = copy(plt.cm.get_cmap('gray'))
    cmap.set_bad(alpha = 0)
    cmap.set_over(col, 1.0)
    im_line = find_boundaries(seg, mode = 'outer')
    im_line = im_line.astype(float)
    im_line[im_line == 0] = np.nan
    clims = (0,0.9)
    extent = (0,seg.shape[1],0,seg.shape[0])
    ax.imshow(im_line, cmap=cmap, clim=clims, interpolation='none')
    return ax


def filter_seg_objects(seg, props, filter):
    seg_new = seg.copy()
    remove_cids = props.loc[props[filter] == 0, ['label','bbox']].values
    for i, (c, b) in enumerate(remove_cids):
        b = eval(b)
        b_sub = seg_new[b[0]:b[2],b[1]:b[3]]
        b_sub = b_sub * (b_sub != c)
        seg_new[b[0]:b[2],b[1]:b[3]] = b_sub
    return seg_new


def recolor_seg(seg, val_dict):
    seg_r = np.zeros(seg.shape)
    seg_r[:] = np.nan
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            id = seg[i,j]
            if id:
                seg_r[i,j] = val_dict[id]
    return seg_r


def get_filename_keys(sample_name, filename_factors, sep='_'):
    keys = []
    for factor in filename_factors:
        match = re.search(r'(?<=' + sep + factor + sep + ')[0-9A-Za-z.]+',
                          sample_name)
        if match:
            level = match.group()
            keys.append(level)
    return keys


def get_nested_dict(key_list, values_list, groupby_key_indices):
    kv_zip = zip(key_list, values_list)
    n_nests = len(groupby_key_indices)
    nest_dict = _group(kv_zip, groupby_key_indices[0])
    if n_nests > 1:
        for k1, v1 in nest_dict.items():
            nest_dict[k1] = _group(v1, groupby_key_indices[1])
            if n_nests > 2:
                for k2, v2 in nest_dict[k1].items():
                    nest_dict[k1][k2] = _group(v2, groupby_key_indices[2])
                    if n_nests > 3:
                        for k3, v3 in nest_dict[k1][k2].items():
                            nest_dict[k1][k2][k3] = _group(v3, groupby_key_indices[3])
    return nest_dict


def _group(kv_pairs, k_index):
    list_dict = defaultdict(list)
    for kv in kv_pairs:
        group = kv[0][k_index]
        list_dict[group].append(kv)
    return list_dict


def measure_regionprops(seg, raw):
    sp_ = regionprops(seg, intensity_image = raw)
    properties=['label','centroid','area','max_intensity','mean_intensity',
                'min_intensity', 'bbox','major_axis_length', 'minor_axis_length',
                'orientation','eccentricity','perimeter']
    df = pd.DataFrame([])
    for p in properties:
        df[p] = [s[p] for s in sp_]
    for j in range(2):
        df['centroid-' + str(j)] = [r['centroid'][j] for i, r in df.iterrows()]
    for j in range(4):
        df['bbox-' + str(j)] = [r['bbox'][j] for i, r in df.iterrows()]
    # regions = regionprops_table(seg, intensity_image = raw,
    #                             properties=['label','centroid','area','max_intensity',
    #                             'mean_intensity','min_intensity', 'bbox',
    #                             'major_axis_length', 'minor_axis_length',
    #                             'orientation','eccentricity','perimeter'])
    # return pd.DataFrame(regions)
    return df
