
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
from sklearn.neighbors import NearestNeighbors



def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    # parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-pf', '--props_fn', dest = 'props_fn', type = str, default = '')
    parser.add_argument('-amf', '--adj_mat_fn', dest = 'adj_mat_fn', type = str, default = '')

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi
    import image_plots as ip

    # # barcode to taxon dict
    # probe_design_fn = probe_design_dir + '/' + config['probe_design_filename']
    # df_probe_design = pd.read_csv(probe_design_fn)
    # dict_bc_tax = dict(zip(df_probe_design['code'], df_probe_design['sci_name']))
    # dict_bc_tax[config['low_int']['code']] = config['low_int']['column']
    # dict_bc_tax[config['high_dist']['code']] = config['high_dist']['column']

    # get props
    props = pd.read_csv(args.props_fn)
    # cell_bcs = props['cell_barcode'].values
    # bcs_unq = np.unique(cell_bcs, return_counts=True)
    # barcodes_sort = [x for _, x in sorted(zip(bcs_unq[1],bcs_unq[0]), reverse=True)]
    # avgint_cols = [str(i) for i in range(config['chan_start'],config['chan_end'])]
    # cell_spec_full = props[avgint_cols].values

    # Get the centroids of the cells and the cell barcodes
    centroids = [list(eval(c)) for c in props['centroid']]
    shp = props.shape[0]
    bc_count = props['cell_barcode'].value_counts()
    bc_sort = bc_count.index
    tax_count = bc_count.values

    # Run nearest neighbor analysis on the centroids
    radius_um = config['adj_network_radius']
    radius_pix = radius_um / config['resolution']
    nbrs = NearestNeighbors(radius=radius_pix).fit(centroids)
    dist, inds = nbrs.radius_neighbors(centroids, return_distance=True)

    # Get the cell barcodes for the nearest neighbors and create adjacency matrix
    n_bcs = bc_sort.shape[0]
    adj_mat = np.zeros((n_bcs,n_bcs))
    for i in range(inds.shape[0]):
        # Get cell index
        cell = inds[i][np.argwhere(dist[i] == 0)][0][0]
        # Get  barcode for cell
        bc_cell = props.loc[cell, 'cell_barcode']
        # Get adj mat index for barcode
        i_cell = np.argwhere(bc_sort == bc_cell)
        # Remove self as adjacency and creaete consistent shape even if there is only one adjacency
        adjs = inds[i][np.argwhere(dist[i] != 0)].reshape(-1).tolist()
        # Add counts to adj mat
        if len(adjs) > 0:
            for a in adjs:
                bc_adj = props.loc[a, 'cell_barcode']
                i_adj = np.argwhere(bc_sort == bc_adj)
                adj_mat[i_cell, i_adj] += 1

    # Save labeled dataframe
    pd.DataFrame(adj_mat, columns=bc_sort, index=tax_count).to_csv(args.adj_mat_fn)

    return

if __name__ == '__main__':
    main()






#####
