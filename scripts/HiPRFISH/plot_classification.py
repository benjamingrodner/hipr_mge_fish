
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
    parser = argparse.ArgumentParser('Plot the classification results')
    # parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-sf', '--seg_fn', dest = 'seg_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-cf', '--classif_fn', dest = 'classif_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-scf', '--seg_col_fn', dest = 'seg_col_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-pscf', '--plot_seg_col_fn', dest = 'plot_seg_col_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-pclf', '--plot_classif_legend_fn', dest = 'plot_classif_legend_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')


    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi
    import image_plots as ip


    ###########
    # Recolor segmentation with classification colors
    print('Coloring classification segmentation')

    props = pd.read_csv(args.classif_fn)
    barcode_color_fn = (
            config['__default__']['PROBE_DESIGN_DIR'] + '/'
            + config['barcode_color_fn']
            )
    dict_label_bbox = dict(props[['label','bbox']].values)
    barcode_color = pd.read_csv(barcode_color_fn)
    dict_bc_col = dict(barcode_color.values)
    dict_bc_col[config['high_dist']['code']] = eval(config['high_dist']['color'])
    dict_bc_col[config['low_int']['code']] = eval(config['low_int']['color'])
    arr_lab_bc = props[['label','cell_barcode']].values
    dict_label_col = {lab: dict_bc_col[bc] for lab, bc in arr_lab_bc}
    # try:
    #     lab_dist = props.loc[props[config['high_dist']['column']], 'label'].values
    #     for l in lab_dist:
    #         dict_label_col[l] = eval(config['high_dist']['color'])
    #     lab_int = props.loc[props[config['low_int']['column']], 'label'].values
    #     for l in lab_int:
    #         dict_label_col[l] = eval(config['low_int']['color'])
    #     dict_bc_col[config['low_int']['code']] = config['low_int']['color']
    #     dict_bc_col[config['high_dist']['code']] = config['high_dist']['color']
    # except:
    #     pass
    seg = np.load(args.seg_fn)
    if not os.path.exists(args.seg_col_fn):
        seg_col = fsi.recolor_image(seg, dict_label_bbox, dict_label_col, threeD=3)
        # seg_col = np.zcarray([])
        np.save(args.seg_col_fn, seg_col)

    # Plot the recolored segmentation
    dpi = np.max(seg.shape) / config['seg_col_plots']['im_inches']
    if not os.path.exists(args.plot_seg_col_fn):
        seg_col = np.load(args.seg_col_fn)
        fig, ax, cbar = ip.plot_image(seg_col, dpi=dpi, **config['seg_col_plots'])
        ip.plt.sca(ax)
        # fig, ax = ip.general_plot()
        out_bn = os.path.splitext(args.plot_seg_col_fn)[0]
        ip.save_png_pdf(out_bn, dpi=dpi)

    # Plot the taxon legend
    probe_design_filename = (config['__default__']['PROBE_DESIGN_DIR'] +
                                '/' + config['probe_design_filename']
                                )
    probe_design = pd.read_csv(probe_design_filename)
    vcounts = props['cell_barcode'].value_counts()
    print('LOOK HERE\n\n', vcounts.index.values,'\n')

    taxon_colors = []
    taxon_names = []
    dict_bc_tax = dict(probe_design[['code','sci_name']].values)
    dict_bc_tax[config['high_dist']['code']] = config['high_dist']['column']
    dict_bc_tax[config['low_int']['code']] = config['low_int']['column']
    for bc in vcounts.index.values:
        col = dict_bc_col[bc]
        col = eval(col) if isinstance(col, str) else col
        taxon_colors.append(col)
        name = dict_bc_tax[bc]
        # name = probe_design.loc[probe_design.code == bc, 'sci_name'].values[0]
        taxon_names.append(name)
    print('LOOK HERE\n\n', taxon_names,'\n')
    ip.taxon_legend(
            taxon_names=taxon_names,
            taxon_colors=taxon_colors,
            taxon_counts=vcounts.values
            )
    out_bn = os.path.splitext(args.plot_classif_legend_fn)[0]
    ip.save_png_pdf(out_bn, dpi=dpi)


    return

if __name__ == '__main__':
    main()






#####
