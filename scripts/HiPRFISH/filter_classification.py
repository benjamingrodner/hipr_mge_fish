
"""
Ben Grodner 2023
De Vlaminck Lab
Cornell University
"""

import os
import re
import sys
import glob
import yaml
import joblib
import skimage
import argparse
import numpy as np
import pandas as pd
from skimage import measure
from sklearn import neighbors as nb

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################################################################
# HiPR-FISH : Image Analysis Pipeline
###############################################################################################################



def main():
    parser = argparse.ArgumentParser('Filter the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    # parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-cf', '--classif_fn', dest = 'classif_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-psrf', '--plot_spec_raw_fn', dest = 'plot_spec_raw_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-pcmf', '--plot_cell_maxint_fn', dest = 'plot_cell_maxint_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-pndf', '--plot_nndist_fn', dest = 'plot_nndist_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-fsf', '--filt_summ_fn', dest = 'filt_summ_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-cff', '--classif_filt_fn', dest = 'classif_filt_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi
    import image_plots as ip

    # Get classifier format
    ref_train_dir = (
            config['output_dir']
            + '/' + config['reference_training']['out_dir']
            )
    spc = config['ref_train_simulations']
    probe_design_filename = (config['__default__']['PROBE_DESIGN_DIR'] +
                                '/' + config['probe_design_filename']
                                )
    # probe_design_basename = os.path.splitext(os.path.basename(probe_design_filename))[0]
    # ref_clf = config['pkl_fmt'].format(ref_train_dir, spc, probe_design_basename, '{}')

    # Load input spectra
    props = pd.read_csv(args.classif_fn)
 
    # Get cell spectra
    avgint_cols = [str(i) for i in range(config['chan_start'],config['chan_end'])]
    avgint = props[avgint_cols].values

    # Get rough classification
    # rc_colnames = ['rough_class_' + str(i) for i in range(config['n_lasers'])]
    # rough_classif = props[rc_colnames]


    ##########
    # Plots illustrating filtering

    # Plot raw spectra
    # if not os.path.exists(args.plot_spec_raw_fn):
    print('Plotting spectra...')
    pdict = config['spectra_plots']
    fig, ax = ip.general_plot(**pdict['axes'])
    ax = fsi.plot_cell_spectra(ax, avgint, pdict['data'])
    xlims = ax.get_xlim()
    ax.plot(
            xlims, [config['spec_min_thresh']]*2,
            lw=pdict['axes']['lw'],
            color=pdict['axes']['col']
            )  # Plot threshold value
    out_bn = os.path.splitext(args.plot_spec_raw_fn)[0]
    ip.save_png_pdf(out_bn, dpi=pdict['dpi'])

    # Plot intensity filter
    # if not os.path.exists(args.plot_spec_norm_fn):
    pdict = config['nndist_plots']
    nn_dists = np.max(avgint, axis=1)
    fig, ax = ip.general_plot(**pdict['axes'])
    ax = fsi.plot_nn_dists(ax, nn_dists, pdict['data'])
    xlims = ax.get_xlim()
    ax.plot(
            xlims, [config['spec_min_thresh']]*2,
            lw=pdict['axes']['lw'],
            color=pdict['axes']['col']
            )  # Plot threshold value
    out_bn = os.path.splitext(args.plot_cell_maxint_fn)[0]
    ip.save_png_pdf(out_bn, dpi=pdict['dpi'])

    # Plot nn dist for all cells
    # if not os.path.exists(args.plot_nndist_fn):
    pdict = config['nndist_plots']
    nn_dists = props['nn_dist'].values
    fig, ax = ip.general_plot(**pdict['axes'])
    ax = fsi.plot_nn_dists(ax, nn_dists, pdict['data'])
    xlims = ax.get_xlim()
    ax.plot(
            xlims, [config['nndist_max_thresh']]*2,
            lw=pdict['axes']['lw'],
            color=pdict['axes']['col']
            )  # Plot threshold value
    out_bn = os.path.splitext(args.plot_nndist_fn)[0]
    ip.save_png_pdf(out_bn, dpi=pdict['dpi'])


    ##########
    # Filter cells
    print('Filtering classification')
    bool_int = np.max(avgint, axis=1) < float(config['spec_min_thresh'])
    bool_dist = props['nn_dist'].values > float(config['nndist_max_thresh'])
    props.loc[bool_int, 'cell_barcode'] = config['low_int']['code']
    props.loc[bool_dist, 'cell_barcode'] = config['high_dist']['code']
    print('LOOK HERE\n\n', props.cell_barcode.value_counts().index.values,'\n')
    props[config['low_int']['column']] = ~bool_int
    props[config['high_dist']['column']] = ~bool_dist
    props_filt = props[~bool_int & ~bool_dist]
    summ = pd.DataFrame(
            [[props.shape[0], np.sum(bool_int),
            np.sum(bool_dist), props_filt.shape[0]]],
            columns=['pre','int_filter','dist_filter','both_filter'])
    summ.to_csv(args.filt_summ_fn, index=False)
    props.to_csv(args.classif_filt_fn, index=False)

    return

if __name__ == '__main__':
    main()






#####
