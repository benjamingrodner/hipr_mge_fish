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
from collections import defaultdict
import javabridge
import bioformats
from tqdm import tqdm
import aicspylibczi as aplc
from scipy.stats import linregress
from skimage.restoration import richardson_lucy




# %% md

# Move to the workdir

# %% codecell
# Absolute path
project_workdir = '/workdir/bmg224/manuscripts/mgefish/code/fig_4/multicolor_phage'

os.chdir(project_workdir)
os.getcwd()  # Make sure you're in the right directory


# %% md

# Define paths

# %% codecell
data_dir = '../../../data/fig_4/multicolor_phage'
output_dir = '../../../outputs/fig_4/multicolor_phage'
pipeline_path = '../..'


# %% codecell
# params
resolution = 0.02995
channel_names = ['termL_DNA_1','termL_RNA','EUB338','termL_DNA_2']

# %% md

# Load specialized modules. Make sure you have the [segmentation pipeline](https://github.com/benjamingrodner/pipeline_segmentation).

# %% codecell
%load_ext autoreload
%autoreload 2

sys.path.append(pipeline_path + '/functions')
import fn_general_use as fgu
import image_plots as ip
import segmentation_func as sf
import fn_hiprfish_classifier as fhc
import fn_spectral_images as fsi

# %% md

# ==============================================================================
# ## Pick a sample
# ==============================================================================

# %% codecell
# get samples
ext = '_mode_airy_Airyscan_Processing_shad_stitch.czi'
filenames = glob.glob(data_dir + '/*' + ext)
sample_names = [re.sub(ext,'',os.path.split(fn)[1]) for fn in filenames]
sample_names

# %% codecell
javabridge.start_vm(class_path=bioformats.JARS)

# %% codecell
# pick sample
i = 2
sn = sample_names[i]
fn = filenames[i]
raw = bioformats.load_image(fn)
raw_chans = [raw[:,:,i] for i in range(raw.shape[2])]

# %% codecell
# metadata
raw_aplc = aplc.CziFile(fn)

for elem in raw_aplc.meta.iter():
    if len(elem.attrib) > 0:
        print(elem.tag, elem.attrib)

# %% codecell

clims = [(0,0.01),(0,0.01),(),(0,0.01)]
ip.subplot_square_images(raw_chans, (raw.shape[2],1), clims=clims)

# %% codecell
# Get rgb overlay on grayscale
im_inches = 20
chan_lims = [(0.01,0.035),(0.005,0.075),(0.01,0.15)]
chan_cols = [(1,1,0),(0,1,1),(1,0,1)]

raw_cell_chan = raw_chans[2]
raw_spot_chans = [raw_chans[i] for i in [0,1,3]]

fig, ax, _ = ip.plot_image(raw_cell_chan, im_inches=im_inches,
        scalebar_resolution=resolution, cmap='gray')


# spot_raw_zoom = raw_shift[0][c[0]:c[0]+w[0],c[1]:c[1]+w[1],1].copy()
rgb_spot_chans = []
for im, lims, col in zip(raw_spot_chans, chan_lims, chan_cols):
    ll, ul = lims
    im_ = im.copy()
    im_ -= np.min(im_)
    im_ /= np.max(im_)
    im_[im_ > ul] = ul
    im_[im_ < ll] = 0
    im_ /= ul
    im_rgb = np.zeros(im_.shape + (4,))
    im_rgb[:,:,3] = im_
    for j, c in enumerate(col):
        if c:
            im_rgb[:,:,j] = im_
    rgb_spot_chans.append(im_rgb)
    ax.imshow(im_rgb)

# %% md

# ==============================================================================
# ## get correlation between colors
# ==============================================================================

# %% codecell
# Divide image into regions,
shape = raw_cell_chan.shape
region_size = 10
rows = np.arange(0,shape[0],region_size)
columns = np.arange(0,shape[1],region_size)
rows.shape, columns.shape

np.max(rows)
np.max(columns)

# %% codecell
# Threshold images
thresh_chans = [0.004,0.02,0.01]
im_inches = 20
clims = [[0,0.01],[],[0,0.05],[],[0,0.025],[]]

raw_spot_chans_thresh = [(im > t) for im, t in zip(raw_spot_chans, thresh_chans)]
im_list = [i for j in zip(raw_spot_chans,raw_spot_chans_thresh) for i in j]

ip.subplot_square_images(im_list,
        (len(raw_spot_chans),2), im_inches=im_inches, clims=clims)

# %% codecell
# Threshold images
c = [6500,6500]
d = [2000,2000]
thresh_chans = [0.003,0.02,0.004]
im_inches = 20
clims = [[0,0.01],[],[0,0.05],[],[0,0.01],[]]

raw_spot_chans_thresh = [(im > t) for im, t in zip(raw_spot_chans, thresh_chans)]
im_list = [i[c[0]: c[0]+d[0], c[1]: c[1]+d[1]] for j in zip(raw_spot_chans,raw_spot_chans_thresh) for i in j]
ip.subplot_square_images(im_list,
        (len(raw_spot_chans),2), im_inches=im_inches, clims=clims)

# %% codecell
# get color intensity correlation across all regions
raw_spot = raw[:,:,[True,True,False,True]]
dict_corr = {'0,1':[[],[]], '0,2':[[],[]],'1,2':[[],[]]}
dict_keys_order = ['0,1','0,2','1,2']
bool_selector = np.array([
        [1,1,0],
        [1,0,1],
        [0,1,1]
        ])
for i in tqdm(range(rows.shape[0] - 1)):
    r_0, r_1 = rows[i], rows[i+1]
    r = raw_spot[r_0:r_1,:,:]

    for j in range(columns.shape[0] - 1):
        c_0, c_1 = columns[j], columns[j+1]
        square = r[:, c_0:c_1, :]
        maxs = [np.max(square[:,:,i]) for i in range(raw_spot.shape[2])]
        bools = np.array([m > t for m, t in zip(maxs, thresh_chans)])

        if sum(bools) == 1:
            selec = bool_selector[bools,:].squeeze().astype(bool)
            keys_towrite = np.array(dict_keys_order)[selec]
        elif sum(bools) > 1:
            keys_towrite = dict_keys_order
        else:
            keys_towrite = []

        for key in keys_towrite:
            inds = list(eval(key))
            dict_corr[key][0].append(maxs[inds[0]])
            dict_corr[key][1].append(maxs[inds[1]])


# %% codecell
# PLot correlations
dims = [1.,1.]
size = 1
col = 'k'
line_col = 'r'
ft=12

names_spot_chan = [channel_names[i] for i in [0,1,3]]
for k in dict_keys_order:
    inds = list(eval(k))
    xlab, ylab = [names_spot_chan[i] for i in inds]
    x,y = dict_corr[k]
    fig, ax = ip.general_plot(dims=dims, ft=ft)
    ax.scatter(x,y,s=size, c=col)

    # coef = np.polyfit(x,y,1)
    # poly1d_fn = np.poly1d(coef)
    # ylims = ax.get_ylim()
    # xlims = ax.get_xlim()
    # plt.plot(xlims, poly1d_fn(xlims), line_col)
    stat_list = linregress(x,y)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    reg_y = [stat_list.slope * x_ + stat_list.intercept for x_ in xlims]
    ax.plot(xlims, reg_y, line_col)
    # xlims = ax.get_xlim()
    # ylims = ax.get_ylim()
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_aspect(1 / ax.get_data_ratio())
    # xticks, yticks = ax.get_xticks(), ax.get_yticks()
    dict_plot = {'xticks':xticks, 'yticks':yticks, 'r_squared':stat_list.rvalue}
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.text(xlims[1] - 0.65*xlims[1], ylims[1] - 0*ylims[1],
    #         'R$^2$=' + str(round(stat_list.rvalue**2, 2)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.1*ylims[1],
    #         'R=' + str(round(stat_list.rvalue, 4)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.2*ylims[1],
    #         'p=' + str(stat_list.pvalue), fontsize=ft)
    # ax.set_aspect('equal')
    # ax.set_xlabel(xlab)
    # ax.set_ylabel(ylab)

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    out_bn = output_dir + '/' + sn + '_corr_' + ylab + '_' + xlab
    # ip.save_png_pdf(out_bn)
    yaml_fn = out_bn + '.yaml'
    # print(dict_plot)
    print('ylab',ylab)
    print('xlab',xlab)
    print('r squared', stat_list.rvalue**2)
    plt.show()
    plt.close()
    # with open(yaml_fn, 'w') as f:
    #     yaml.dump(dict_plot, f)

# %% codecell
# Save correlations
dims = [0.5,0.5]
size = 1
col = 'k'
line_col = 'r'
ft=6

names_spot_chan = [channel_names[i] for i in [0,1,3]]
for k in dict_keys_order:
    inds = list(eval(k))
    xlab, ylab = [names_spot_chan[i] for i in inds]
    x,y = dict_corr[k]
    fig, ax = ip.general_plot(dims=dims, ft=ft)
    ax.scatter(x,y,s=size, c=col)

    # coef = np.polyfit(x,y,1)
    # poly1d_fn = np.poly1d(coef)
    # ylims = ax.get_ylim()
    # xlims = ax.get_xlim()
    # plt.plot(xlims, poly1d_fn(xlims), line_col)
    stat_list = linregress(x,y)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    reg_y = [stat_list.slope * x_ + stat_list.intercept for x_ in xlims]
    ax.plot(xlims, reg_y, line_col)
    # xlims = ax.get_xlim()
    # ylims = ax.get_ylim()
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_aspect(1 / ax.get_data_ratio())
    # xticks, yticks = ax.get_xticks(), ax.get_yticks()
    dict_plot = {'xticks':xticks, 'yticks':yticks, 'r_squared':stat_list.rvalue}
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.text(xlims[1] - 0.65*xlims[1], ylims[1] - 0*ylims[1],
    #         'R$^2$=' + str(round(stat_list.rvalue**2, 2)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.1*ylims[1],
    #         'R=' + str(round(stat_list.rvalue, 4)), fontsize=ft)
    # ax.text(xlims[1] - 0.25*xlims[1], ylims[1] - 0.2*ylims[1],
    #         'p=' + str(stat_list.pvalue), fontsize=ft)
    # ax.set_aspect('equal')
    # ax.set_xlabel(xlab)
    # ax.set_ylabel(ylab)
    print(ax.get_xticks(), ax.get_yticks())
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    out_bn = output_dir + '/' + sn + '_corr_' + ylab + '_' + xlab
    # ip.save_png_pdf(out_bn)
    plt.show()
    plt.close()




# %% codecell
def generate_point_spread_function(shape=(11, 11), sigma=2.0):
    x, y = np.meshgrid(np.arange(-shape[1]//2, shape[1]//2+1), np.arange(-shape[0]//2, shape[0]//2+1))
    psf = np.exp(-(x**2 + y**2)/(2.0*sigma**2))
    psf /= np.sum(psf)
    return psf



# %% codecell
# Pick a psf
shape=(50,50)
sigma=6
psf = generate_point_spread_function(shape=shape,sigma=sigma)

# %% codecell
c = [7825,7100]
d = [50,50]
clims = [(0,0.01),()]
im_inches=10
spot_0 = raw_spot_chans[2][c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
spot_list = [spot_0, psf]
ip.subplot_square_images(spot_list, (1,2), clims=clims, im_inches=im_inches)

# %% codecell
c = [7500,7000]
d = [500,500]
clims = [(0,0.01),(0,0.03)]
spot_2_zoom = raw_spot_chans[2][c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
spot_2_zoom_deconv = richardson_lucy(spot_2_zoom, psf)
spot_list = [spot_2_zoom, spot_2_zoom_deconv]
ip.subplot_square_images(spot_list, (1,2), clims=clims, im_inches=im_inches)

# %% codecell
c = [6500,6500]
d = [2000,2000]
clims = [(0,0.01),(0,0.03)]
spot_2_zoom = raw_spot_chans[2][c[0]: c[0]+d[0], c[1]: c[1]+d[1]]
spot_2_zoom_deconv = richardson_lucy(spot_2_zoom, psf)
spot_list = [spot_2_zoom, spot_2_zoom_deconv]
ip.subplot_square_images(spot_list, (1,2), clims=clims, im_inches=im_inches)
