
"""
Ben Grodner 2023
De Vlaminck Lab
Cornell University

Purpose: Used in pipeline for registering HiPRFISH images with MeGAFISH images
to correct x-y shifts. Upsamples HiPRFISH images so the resolution matches
MeGAFISH images, then uses phase cross correlation to overlay the two images.
"""

################################################################################
# Imports
################################################################################

import os
# import re
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from random import sample
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

################################################################################
# Functions
################################################################################


################################################################################
# Script
################################################################################


def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-spf', '--spot_props_fns', dest = 'spot_props_fns', nargs = '+', default = [])
    parser.add_argument('-cpf', '--cell_props_fn', dest = 'cell_props_fn', type=str)
    parser.add_argument('-cbf', '--cell_barcode_fn', dest = 'cell_barcode_fn', type=str)
    parser.add_argument('-dtcs', '--dict_tax_count_sim_fns', dest = 'dict_tax_count_sim_fns', nargs = '+', default = [])
    parser.add_argument('-tpf', '--tax_prob_fns', dest = 'tax_prob_fns', nargs = '+', default = [])

    # parser.add_argument('-hmr', '--hipr_max_resize_fn', dest = 'hipr_max_resize_fn', type = str, default = '', help = 'Input normalized single cell spectra filename')

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    ### Setup
    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)
    with open(config['config_hipr'], 'r') as f:
        config_hipr = yaml.safe_load(f)
    with open(config['config_mega'], 'r') as f:
        config_mega = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config_mega['functions_path'])
    import image_plots as ip
    import segmentation_func as sf

    # Cell coords
    cell_props = pd.read_csv(args.cell_props_fn)
    cell_coords = [eval(c) for c in cell_props['centroid'].values]
    cell_coords = np.rint(cell_coords).astype(np.int64)
    # add cell barcodes
    cell_barcode_df = pd.read_csv(args.cell_barcode_fn)
    cell_props = cell_props.merge(cell_barcode_df[['label','cell_barcode']], how='left', on='label')
    dict_index_tax = {i:int(bc) for i,bc in cell_props['cell_barcode'].iteritems()}

    # Knn object
    r_pix = config['assoc_radius'] / config_mega['resolution']
    nbrs = NearestNeighbors(radius=r_pix, algorithm='kd_tree').fit(cell_coords)

    # convert barcode to taxon name
    probe_design_fn = (config_hipr['__default__']['PROBE_DESIGN_DIR']
            + '/' + config_hipr['probe_design_filename'])
    probe_design = pd.read_csv(probe_design_fn)
    dict_bc_sciname = dict(probe_design[['code','sci_name']].values)
    dict_bc_sciname[config_hipr['low_int']['code']] = config_hipr['low_int']['column']
    dict_bc_sciname[config_hipr['high_dist']['code']] = config_hipr['high_dist']['column']

    # convert barcode to color
    col_df_fn = (config_hipr['__default__']['PROBE_DESIGN_DIR']
            + '/' + config_hipr['barcode_color_fn'])
    col_df = pd.read_csv(col_df_fn)
    dict_bc_col = dict(col_df[['barcode','color']].values)
    dict_bc_col[config_hipr['low_int']['code']] = config_hipr['low_int']['color']
    dict_bc_col[config_hipr['high_dist']['code']] = config_hipr['high_dist']['color']

    for spot_props_fn, dict_tax_count_sim_fn, tax_prob_fn in zip(
                args.spot_props_fns,
                args.dict_tax_count_sim_fns,
                args.tax_prob_fns
                ):
        # Get Spot coords
        spot_props = pd.read_csv(spot_props_fn)
        n = spot_props.shape[0]
        spot_coords = [eval(c) for c in spot_props['centroid'].values]
        spot_coords = np.rint(spot_coords).astype(np.int64)

        # Get nearest neighbors
        nn_distances, nn_indices = nbrs.radius_neighbors(spot_coords)

        # Convert neaeres neighbro indices to taxa
        nn_bcs = [[dict_index_tax[i] for i in ind] for ind in nn_indices]
        # Compress each list of neighbors to unique taxa
        nn_bcs_unq = np.concatenate([np.unique(bcs) for bcs in nn_bcs])
        # Count the number of each taxa
        nn_tax_bc, nn_tax_counts = np.unique(nn_bcs_unq, return_counts=True)
        # dict_tax_assoc = defaultdict(list)
        # for t, c in zip(nn_tax_bc, nn_tax_counts):
        #     dict_tax_assoc[dict_bc_sciname[t]].append(c/n)

        # Count randomized species associations
        # Get cell nearest neighbors
        cell_nn_distances, cell_nn_indices = nbrs.radius_neighbors(cell_coords)
        # Convert between cell index and neighbors
        dict_index_nnbc = {
                i:[dict_index_tax[j] for j in nn]
                for i, nn in zip(cell_props.index.values, cell_nn_indices)
                }

        # Randomly sample indexes of cells as proxy for spot location
        n=config['n_assoc_simulations']
        dict_tax_count_sim = defaultdict(list)
        for i in range(n):
            spot_sample = sample(
                    cell_props.index.values.tolist(),
                    len(spot_coords)
                    )
            nn_bcs_ = [dict_index_nnbc[i] for i in spot_sample]
            nn_bcs_unq_ = np.concatenate([np.unique(bcs) for bcs in nn_bcs_])
            nn_tax_bc_, nn_tax_counts_ = np.unique(nn_bcs_unq_, return_counts=True)
            for k, v in zip(nn_tax_bc_, nn_tax_counts_):
                dict_tax_count_sim[k].append(v)
        with open(dict_tax_count_sim_fn, 'w') as f:
            yaml.dump(dict_tax_count_sim, f)

        # calculate probability
        dict_sciname_prob = {}
        dict_bc_ltgt = {}
        tax_assoc = []
        for bc_tax, true_count in zip(nn_tax_bc, nn_tax_counts):
            sim_counts = np.array(dict_tax_count_sim[bc_tax])
            sim_counts_mean = np.mean(sim_counts)
            if true_count > sim_counts_mean:
                r_ = sim_counts[sim_counts >= true_count].shape[0]
                sgn = 1
            else:
                r_ = sim_counts[sim_counts <= true_count].shape[0]
                sgn = -1
            p_ = r_ / n
            # Add to assoc dict
            scm_frac = sim_counts_mean / n
            true_frac = true_count / n
            sci_name = dict_bc_sciname[bc_tax]
            tax_assoc.append([sci_name, true_frac, scm_frac, p_, sgn])
        columns = ['sci_name','obs_frac_spots_assoc','sim_mean_frac_spots_assoc', 'propability','sign']
        tax_assoc_df = pd.DataFrame(tax_assoc, columns=columns)
        tax_assoc_df.to_csv(tax_prob_fn)

        # Plot
        alpha=1
        true_col='k'
        true_lw=2
        lw=1
        dims=(2,1)
        ft=7
        nbins=100
        # bin_scaling=2
        for bc_tax, true_count in zip(nn_tax_bc, nn_tax_counts):
            target_taxon = dict_bc_sciname[bc_tax]
            col = dict_bc_col[bc_tax]
            color = eval(col) if isinstance(col, str) else col

            # Counts from simulation
            rand_counts = dict_tax_count_sim[bc_tax]
            # Get fraction of total spots
            rand_frac = np.array(rand_counts) / spot_coords.shape[0]
            # plot distribution of spot assignment
            nbins = np.unique(rand_frac).shape[0] * 2
            bins = np.linspace(np.min(rand_frac), np.max(rand_frac), nbins)
            # bins = np.arange(np.min(rand_frac), np.max(rand_frac))
            # hist, bin_edges = np.histogram(rand_frac)
            hist, bin_edges = np.histogram(rand_frac, bins=bins)
            x_vals = ip.get_line_histogram_x_vals(bin_edges)
            fig, ax = ip.general_plot(dims=dims, ft=ft, lw=lw)
            ax.plot(x_vals, hist, color=color)
            ax.fill_between(
                    x_vals,
                    np.zeros((x_vals.shape[0],)),
                    hist,
                    alpha=alpha,
                    color=color
                    )
            # PLot expected value
            ylims = ax.get_ylim()
            rand_count_mean = np.mean(rand_frac)
            ax.plot([rand_count_mean]*2, [0,0.75*ylims[1]], 'grey', lw=lw)
            # plot location of actual assignment number
            true_frac = true_count / spot_coords.shape[0]
            ax.plot([true_frac]*2, [0,0.75*ylims[1]], color=true_col, lw=true_lw)
            # ax.set_xlabel('MGE association counts')
            # ax.set_ylabel('Frequency')
            # ax.set_title(target_taxon)
            output_basename = os.path.splitext(dict_tax_count_sim_fn)[0] + '_{taxon}'
            ip.save_png_pdf(output_basename.format(taxon=target_taxon))
            plt.show()
            plt.close()

    return

if __name__ == '__main__':
    main()






#####
