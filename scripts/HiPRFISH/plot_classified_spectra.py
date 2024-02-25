
"""
Ben Grodner 2023
De Vlaminck Lab
Cornell University

HiPRFISH processing pipeline script

Plot the raw spectra as grouped by the classifier
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd



def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    # parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-pf', '--props_fn', dest = 'props_fn', type = str, default = '', help = 'Input classified spectra table')
    parser.add_argument('-sn', '--sample_name', dest = 'sample_name', type = str, default = '', help = '')
    parser.add_argument('-od', '--out_dir', dest = 'out_dir', type = str, default = '', help = 'Directory to write spectra plots for this image')
    parser.add_argument('-cf', '--complete_fn', dest = 'complete_fn', type = str, default = '', help = 'File to tell snakemake its done')

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi
    import image_plots as ip

    # barcode to color dict
    probe_design_dir = config['__default__']['PROBE_DESIGN_DIR']
    df_bc_col = pd.read_csv(probe_design_dir + '/' + config['barcode_color_fn'])
    colors = df_bc_col['color'].apply(eval)
    dict_bc_col = dict(zip(df_bc_col['barcode'], colors))
    dict_bc_col[config['low_int']['code']] = eval(config['low_int']['color'])
    dict_bc_col[config['high_dist']['code']] = eval(config['high_dist']['color'])
    # barcode to taxon dict
    probe_design_fn = probe_design_dir + '/' + config['probe_design_filename']
    df_probe_design = pd.read_csv(probe_design_fn)
    dict_bc_tax = dict(zip(df_probe_design['code'], df_probe_design['sci_name']))
    dict_bc_tax[config['low_int']['code']] = config['low_int']['column']
    dict_bc_tax[config['high_dist']['code']] = config['high_dist']['column']
    # get props
    props = pd.read_csv(args.props_fn)
    cell_bcs = props['cell_barcode'].values
    bcs_unq = np.unique(cell_bcs, return_counts=True)
    barcodes_sort = [x for _, x in sorted(zip(bcs_unq[1],bcs_unq[0]), reverse=True)]
    avgint_cols = [str(i) for i in range(config['chan_start'],config['chan_end'])]
    cell_spec_full = props[avgint_cols].values


    # PLot the spectra from the cells as classified
    out_fmt = os.path.splitext(args.out_dir + '/' + config['plot_spec_classif_fmt'])[0]
    kwargs = config['spectra_plots']['data']
    print('LOOK HERE\n\n', barcodes_sort,'\n')
    for bc in barcodes_sort:
        bool = (cell_bcs == bc)
        spec_ = cell_spec_full[bool,:]
        dict_label_col = {}
        tax = dict_bc_tax[bc]
        print(tax)
        fig, ax = ip.general_plot(**config['spectra_plots']['axes'])
        kwargs['color'] = dict_bc_col[bc]
        fsi.plot_cell_spectra(ax, spec_, kwargs)
        ax.set_title(tax)
        out_bn = out_fmt.format(sample_name=args.sample_name, taxon=tax)
        ip.save_png_pdf(out_bn)

    with open(args.complete_fn, 'w') as f:
        f.write('plot_spec_classif_complete')

    return

if __name__ == '__main__':
    main()






#####
