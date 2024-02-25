import glob
import sys
import os
# import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
# import re
# import joblib
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict
import numba as nb
# import javabridge
# import bioformats
import aicspylibczi as aplc
import argparse



# =============================================================================

def reshape_aics_image(m_img):
    '''
    Given an AICS image with just XY and CHannel,
    REshape into shape (X,Y,C)
    '''
    img = np.squeeze(m_img)
    img = np.transpose(img, (1,2,0))
    return img


def register_channels(m_raws, max_shift):
    raws_max = [np.sum(im, axis=2) for im in m_raws]
    raws_max_norm = [im / np.max(im) for im in raws_max]
    shift_vectors = fsi._get_shift_vectors(raws_max_norm)
    return fsi._shift_images(m_raws, shift_vectors, max_shift=max_shift)

def get_smooth(m_raws_shift, sigma):
    raws_smooth = []
    for im in m_raws_shift:
        im_smooth = np.empty(im.shape)
        for i in range(im.shape[2]):
            im_smooth[:,:,i] = gaussian_filter(im[:,:,i], sigma=sigma)
        raws_smooth.append(im_smooth)
    return raws_smooth

def get_mask(stack_smooth, max_thresh, out_bn, config):
    max_sub = np.max(stack_smooth, axis=2)
    y = np.sort(np.ravel(max_sub))
    x = np.arange(y.shape[0])
    fig, ax = ip.general_plot()
    ax.plot(x,y)
    xlims = ax.get_xlim()
    ax.plot(xlims, [max_thresh]*2)
    out_bn_plot = out_bn + '_thresh_plot'
    ip.save_png_pdf(out_bn_plot)
    print('Wrote:', out_bn_plot + '.png')

    mask = max_sub > max_thresh

    max_masked = max_sub * mask
    clims = config['mask_plots']['clims']
    im_inches = config['mask_plots']['im_inches']
    ip.subplot_square_images([max_sub, mask, max_masked], (1,3), 
                            clims=clims, im_inches=im_inches)
    out_bn_im = out_bn + '_images'
    ip.save_png_pdf(out_bn_im)
    print('Wrote:', out_bn_im + '.png')
    
    return mask


def get_reference_spectra(barcodes, bc_len, config):
    ref_dir = config['hipr_ref_dir']
    fmt = config['ref_files_fmt']
    if bc_len == 5:
        barcodes_str = [str(bc).zfill(5) for bc in barcodes]
        # barcodes_str = [str(bc).zfill(7) for bc in barcodes]
        barcodes_10bit = [bc[0] + '0' + bc[1] + '0000' + bc[2:] for bc in barcodes_str]
        # barcodes_10bit = [bc[0] + '0' + bc[1:4] + '00' + bc[4:] for bc in barcodes_str]
        barcodes_b10 = [int(str(bc),2) for bc in barcodes_10bit]
        st = config['ref_chan_start'] + config['chan_start']
        en = config['ref_chan_start'] + config['chan_end']
        ref_avgint_cols = [i for i in range(st,en)]

        ref_spec = []
        for bc in barcodes_b10:
            fn = ref_dir + '/'+ fmt.format(bc)
            ref = pd.read_csv(fn, header=None)
            ref = ref[ref_avgint_cols].values
            ref_spec.append(ref)
    return ref_spec

def sum_normalize_ref(ref_spec):
    ref_sum_norm = []
    for r in ref_spec:
        r_ = r - np.min(r, axis=1)[:,None]
        ref_sum_norm.append(r_ / np.sum(r_, axis=1)[:,None])
    return [np.mean(r, axis=0) for r in ref_sum_norm]


# Get area under spec curves 
def get_spec_curves_area(spec_pix):
    spec_zoom_areas = []
    for sp in spec_pix:
        spec_zoom_areas.append(np.sum(sp))
    return spec_zoom_areas


# Adjust allfluor spectrum to subtract the background
def adjust_allfluor_spectrum(allfluor_spec_norm, spec, m, spec_areas, sig):
    ar_mean = np.mean(spec_areas)
    ar_std = np.std(spec_areas)
    thresh = ar_mean - sig*ar_std
    bool_ar = np.array(spec_areas) < thresh
    allfluor_spec_adj = allfluor_spec_norm * thresh
    fig, ax = ip.general_plot(dims=config['allfluor']['dims'])
    fsi.plot_cell_spectra(ax, spec[bool_ar,:], {'lw':0.1,'alpha':0.1,'color':'r'})
    fsi.plot_cell_spectra(ax, allfluor_spec_adj[None,:], {'lw':1,'alpha':1,'color':'r'})
    out_bn = config['allspec_plot_dir'] + '/' + args.sn + '_M_' + str(m) + '_allspec'
    ip.save_png_pdf(out_bn)
    print('Wrote:', out_bn + '.png')
    return allfluor_spec_adj


def remove_allfluor_barcode(spec_pix, spec_gem, scale_gem):
    # spec_pix_ = spec_pix - np.min(spec_pix, axis=1)[:,None]
    # spec_pix_norm = spec_pix_ / np.sum(spec_pix_, axis=1)[:,None]
    # spec_pix_gem = spec_pix_norm - spec_gem[None,:]*scale_gem
    spec_pix_gem = spec_pix - spec_gem[None,:]
    spec_pix_gem[spec_pix_gem < 0] = 0
    return spec_pix_gem


def run_matrix_multiply(spec_pix_gem, weights_sum_norm):
    weights_t = np.array(weights_sum_norm).T
    return np.matmul(spec_pix_gem, weights_t)

def get_present_filter(probe_design, nlas):
    present_col = probe_design['laser_present'].values
    present_str = [str(lp).zfill(nlas) for lp in present_col]
    present_arr = np.array([[int(l) for l in lp] for lp in present_str])
    return present_arr.T

def remove_possibilities_laserpresent(raws_smooth, mask, classif_mat, present_filter):
    las_max = [np.max(im, axis=2) for im in raws_smooth]
    # Get arrays for pixels
    las_max_stack = np.dstack(las_max)
    las_max_pix = las_max_stack[mask]
    las_max_pix_norm = las_max_pix / np.max(las_max_pix, axis=1)[:,None]
    # Define filter
    las_frac_thresh = config['laser_present_thresholds']  
    las_max_present = las_max_pix_norm > las_frac_thresh
    # present_filter = np.array([
    #         [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #         [0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,0]
    #         ])
    # Run filtering
    classif_adj_mat = np.matmul(las_max_present, present_filter)
    classif_adj_mat_bool = classif_adj_mat == np.max(classif_adj_mat, axis=1)[:,None]
    return classif_mat * classif_adj_mat_bool

def pick_maximum_weight(classif_mat_adj, sci_names):
    classifs_index = np.argmax(classif_mat_adj, axis=1)
    return np.array([sci_names[i] for i in classifs_index])


# Filter dim spectra 
def filter_dim_spectra(classifs, spectra_adj, thresh):
    spec_zoom_max = np.max(spectra_adj, axis=1)
    spec_zoom_max_norm = spec_zoom_max / np.max(spec_zoom_max)
    bool_max = spec_zoom_max_norm < thresh
    classifs[bool_max] = 'None'
    return classifs


def save_classifs(classifs, pix_inds, out_fn):
    arr_classifs = np.hstack([pix_inds[0][:,None], pix_inds[1][:,None], classifs[:,None]])
    df_classifs = pd.DataFrame(arr_classifs, columns=['x','y','sci_name'])
    df_classifs.to_csv(out_fn, index=False)
    print('Wrote:', out_fn)


def matrix_classify(m_raws, m, args, config, vals, spec_plots):
    # Register shifts
    raw_shift = register_channels(m_raws, config['max_shift'])

    # HOusekeeping on raws after shifting
    for i, (r,l) in enumerate(zip(raw_shift, config['lasers'])):
        # collect max values for RBG
        r_sum = np.sum(r, axis=2)
        max_ = np.max(r_sum)
        r_max = vals['dict_r_max'][i]
        r_max = max_ if max_ > r_max else r_max
        vals['dict_r_max'][i] = r_max
        # Min vals for RGB
        min_ = np.min(r_sum)
        r_min = vals['dict_r_min'][i]
        r_min = min_ if min_ > r_min else r_min
        vals['dict_r_min'][i] = r_min   
        # Save shifted images 
        out_fn = args.shift_dir + '/' + args.sn + '_M_' + str(m) + '_' + str(l) + '.npy'
        np.save(out_fn, r)
        print('Wrote:', out_fn)

    # Collect max and min values for sum image         
    stack = np.dstack(raw_shift)
    stack_sum = np.sum(stack, axis=2)
    s_max = np.max(stack_sum)
    s_min = np.min(stack_sum)
    sum_max = vals['sum_max']
    sum_min = vals['sum_min']
    vals['sum_max'] = s_max if s_max > sum_max else sum_max
    vals['sum_min'] = s_min if s_min < sum_min else sum_min

    # Gaussian smoothing for spectra
    raw_smooth = get_smooth(raw_shift, config['sigma'])
    stack_smooth = np.dstack(raw_smooth)

    # Mask background
    mask_out_bn = args.mask_dir + '/' + args.sn + '_M_' + str(m) + '_mask'
    mask = get_mask(stack, args.mask_thresh, mask_out_bn, config)
    spec_pix = stack_smooth[mask > 0]
    pix_inds = np.where((mask > 0))

    # Get reference spectra
    probe_design_dir = config['probe_design_dir']
    probe_design_fn = probe_design_dir + '/' + config['probe_design_filename']
    probe_design = pd.read_csv(probe_design_fn)
    barcodes = probe_design['code'].unique()
    barcode_length = len(str(np.max(barcodes)))
    ref_spec = get_reference_spectra(barcodes, barcode_length, config)
    sci_names = [probe_design.loc[probe_design['code'] == bc,'sci_name'].unique()[0] 
                for bc in barcodes]
    weights_sum_norm = sum_normalize_ref(ref_spec)


    # Correct for background staining with all fluors
    allfluor_bc = [int(''.ljust(barcode_length, '1'))]
    allfluor_spec = get_reference_spectra(allfluor_bc, barcode_length, config)
    allfluor_spec_norm = sum_normalize_ref(allfluor_spec)[0]
    spec_pix_areas = get_spec_curves_area(spec_pix)
    allfluor_spec_adj = adjust_allfluor_spectrum(
        allfluor_spec_norm, spec_pix, m, spec_pix_areas, 
        sig=config['allfluor']['sigma']
    )
    spec_pix_gem = remove_allfluor_barcode(spec_pix, allfluor_spec_adj, args.allfluor)
    
    # Classify pixels
    classif_mat = run_matrix_multiply(spec_pix_gem, weights_sum_norm)
    present_filter = get_present_filter(probe_design, len(config['lasers']))
    classif_mat_adj = remove_possibilities_laserpresent(raw_smooth, mask, 
            classif_mat, present_filter)
    classifs = pick_maximum_weight(classif_mat_adj, sci_names)
    classifs = filter_dim_spectra(classifs, spec_pix_gem, 
                                  thresh=config['dim_spec_filt'])
    out_fn = args.classif_dir + '/' + args.sn + '_M_' + str(m) +  '_classification.csv'
    save_classifs(classifs, pix_inds, out_fn)

    # Add to the plots
    labs = np.unique(classifs)
    for l in labs:
        bool_l = (np.array(classifs) == l)
        spg_sub = spec_pix_gem[bool_l,:]
        sp_sub = spec_pix[bool_l,:]

        for spec_, t in zip([sp_sub, spg_sub],['raw','allfluorsubtracted']):
            shp = spec_.shape[0]
            n = config['spec_plots']['num']
            n = n if n < shp else shp
            inds = np.random.choice(shp, n, replace=False)
            spec_rand = spec_[inds,:]

            try:
                fig, ax = spec_plots[l][t]
            except:
                fig, ax = ip.general_plot(spec_plots['dims'])
                ax.set_title(l + ' ' + t)
            fsi.plot_cell_spectra(ax, spec_rand, config['spec_plots']['kwargs'])
            spec_plots[l][t] = [fig, ax]
            
    return(vals, spec_plots)
    



# =============================================================================


parser = argparse.ArgumentParser(description='Script Description')

# Add arguments to the parser

parser.add_argument('-cfn', '--config_fn', dest ='config_fn', 
                    type = str, help = 'Path to configureaiton file')
parser.add_argument('-sn', '--sample_name', dest ='sn', 
                    type = str, help = 'Sample name wildcard')
parser.add_argument('-sd', '--shift_dir', dest ='shift_dir', 
                    type = str, help = 'Directory to save channel shifts.')
parser.add_argument('-mt', '--mask_thresh', dest ='mask_thresh', 
                    type = float, help = 'Value to remove background.')
parser.add_argument('-md', '--mask_dir', dest ='mask_dir', 
                    type = str, help = 'Directory to save mask plots.')
parser.add_argument('-af', '--allfluor', dest ='allfluor', 
                    type = float, help = 'Amount of allfluor spectrum to subtract.')  
parser.add_argument('-cd', '--classif_dir', dest ='classif_dir', 
                    type = str, help = 'Directory to save pixel classification.')    
parser.add_argument('-mf', '--minmax_fn', dest ='minmax_fn', 
                    type = str, help = 'Filename to save min and max values table.')    
parser.add_argument('-spd', '--spec_plot_dir', dest ='spec_plot_dir', 
                    type = str, help = 'Filename to save min and max values table.')    

# Parse the command-line arguments
args = parser.parse_args()

# Load config file
with open(args.config_fn, 'r') as f:
    config = yaml.safe_load(f)

# Special imports 
sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
import image_plots as ip
import fn_spectral_images as fsi


# =============================================================================


def main():
    """
    Script to classify hiprfish images using a matrix based classifier.

    Usage:
        python sc_matrix_classify.py input_file output_file [--option1 OPTION1] [--option2 OPTION2]

    Example:
        python script_name.py data.csv processed_data.csv --option1 10 --option2 "value"
    """

    # initialize direcotries
    for d in [args.shift_dir, args.mask_dir, args.classif_dir, args.spec_plot_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Get the raw data paths 
    data_dir = config['data_dir']
    raw_regex = data_dir + '/' + args.sn + config['laser_regex']  # Set in config
    raw_fns = sorted(glob.glob(raw_regex))

    # Get image shape information
    czi = aplc.CziFile(raw_fns[0])
    dims_shape = czi.get_dims_shape()[0]
    z_size = dims_shape['Z'][1]
    dims_str = czi.dims
    if 'S' in dims_str:
        s_size = dims_shape['S'][1]
    else:
        s_size = 0

    # Holders for max and min values for RGB
    vals = {}
    vals['dict_r_max'] = {0:0,1:0,2:0}
    vals['dict_r_min'] = {0:10^15,1:10^15,2:10^15}
    # Holders for sum max and min values
    vals['sum_min'] = 10^5
    vals['sum_max'] = 0

    # Image plots holder
    spec_plots = defaultdict(dict)

    # If the image is a mosaic
    if czi.is_mosaic():
        # Get mosaic size
        mosaic_size = dims_shape['M'][1]
        for m in range(mosaic_size):
            m_raws = []
            for fn in raw_fns:
                czi = aplc.CziFile(fn)
                m_img, shp = czi.read_image(M=m)
                img = reshape_aics_image(m_img)
                m_raws.append(img)
            vals, spec_plots = matrix_classify(m_raws, m, args, config, vals, spec_plots)
        
    # If the image is a z stack
    elif z_size > 1:
        mosaic_size = z_size
        for z in range(z_size):
            z_raws = []
            for fn in raw_fns:
                czi = aplc.CziFile(fn)
                z_img, shp = czi.read_image(Z=z)
                img = reshape_aics_image(z_img)
                z_raws.append(img)
            vals, spec_plots = matrix_classify(z_raws, z, args, config, vals, spec_plots)

    # If the image has multiple scenes
    elif s_size > 1:
        mosaic_size = s_size
        for s in range(s_size):
            s_raws = []
            for fn in raw_fns:
                czi = aplc.CziFile(fn)
                s_img, shp = czi.read_image(S=s)
                img = reshape_aics_image(s_img)
                s_raws.append(img)
            vals, spec_plots = matrix_classify(s_raws, s, args, config, vals, spec_plots)

    # If the image has only one tile
    else:
        m = 0
        raws = []
        for fn in raw_fns:
            czi = aplc.CziFile(fn)
            img_, shp = czi.read_image()
            img = reshape_aics_image(img_)
            raws.append(img)
        vals, spec_plots = matrix_classify(raws, m, args, config, vals, spec_plots)

    # Save min and max values 
    dict_df = {k:[vals['dict_r_max'][k], vals['dict_r_min'][k]] 
                for k in vals['dict_r_max'].keys()}
    out_df = pd.DataFrame(dict_df)
    # out_df = pd.DataFrame(vals['dict_r_max']).append(vals['dict_r_min'], ignore_index=True)
    out_df['sum'] = [vals['sum_max'], vals['sum_min']]
    out_df.index = ['max','min']
    out_df.to_csv(args.minmax_fn)

    # Save spectral plots 
    for l, tp in spec_plots.items():
        for t, (fig, ax) in tp.items():
            plt.figure(fig)
            out_bn = args.spec_plot_dir + '/' + args.sn + '_' + l + '_' + t + '_spectra'
            ip.save_png_pdf(out_bn)


if __name__ == "__main__":
    main()
