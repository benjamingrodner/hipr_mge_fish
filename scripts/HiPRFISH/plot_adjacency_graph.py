
"""
Ben Grodner 2023
De Vlaminck Lab
Cornell University

HiPRFISH processing pipeline script

"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    # parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-amf', '--adj_mat_fn', dest = 'adj_mat_fn', type = str, default = '')
    parser.add_argument('-pag', '--plot_adj_graph_fn', dest = 'plot_adj_graph_fn', type = str, default = '')

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi
    import image_plots as ip

    # get adj mat
    adj_mat_df = pd.read_csv(args.adj_mat_fn, index_col=0)
    adj_mat = adj_mat_df.values
    bc_sort = adj_mat_df.columns.values.astype(float).astype(int)
    tax_count = adj_mat_df.index.values

    # convert adj matrix to networkx object
    if not adj_mat.shape[0] == adj_mat.shape[1]: print(adj_mat_df, adj_mat)
    adj_mat_norm = adj_mat / np.max(adj_mat)
    adj_mat_norm.shape
    g = nx.from_numpy_matrix(adj_mat_norm)

    # Get graph colors and labels
    probe_color_fn = config['__default__']['PROBE_DESIGN_DIR'] + '/' + config['barcode_color_fn']
    probe_color = pd.read_csv(probe_color_fn)
    bc_color_dict = dict(zip(probe_color['barcode'],probe_color['color']))
    bc_color_dict[config['low_int']['code']] = config['low_int']['color']
    bc_color_dict[config['high_dist']['code']] = config['high_dist']['color']
    tax_colors =[]
    for bc in bc_sort:
        col = bc_color_dict[bc]
        if isinstance(col, str):
            col = eval(col)
        tax_colors.append(col)
    # tax_colors = [eval(bc_color_dict[bc]) for bc in bc_sort]

    probe_design_filename = config['__default__']['PROBE_DESIGN_DIR'] + '/' + config['probe_design_filename']
    probe_design = pd.read_csv(probe_design_filename)
    bc_sciname_dict = dict(zip(probe_design['code'], probe_design['sci_name']))
    bc_sciname_dict[config['low_int']['code']] = config['low_int']['column']
    bc_sciname_dict[config['high_dist']['code']] = config['high_dist']['column']
    tax_names_sort = [bc_sciname_dict[bc] for bc in bc_sort]


    # graph edge width is interaction frequency
    affinities  = {}
    for edge in g.edges():
        affinities[edge] = adj_mat_norm[edge[0], edge[1]]
    nx.set_edge_attributes(g, affinities, 'my_weight')

    # Plot properties
    prps = config['adj_graph_plots']

    # get layout
    # pos = nx.kamada_kawai_layout(g, weight='my_weight')
    pos = nx.spring_layout(g, weight='my_weight')
    edge_wts = [prps['edge_scale'] * adj_mat_norm[edge[0], edge[1]]
            for edge in g.edges()]

    # node size is species frequency
    node_szs = prps['node_scale'] * (tax_count / np.max(tax_count))
    n_colors = tax_colors
    tax_names_dict = dict(zip(np.arange(bc_sort.shape[0]), tax_names_sort))

    # Plotting
    plt.figure(figsize=prps['dims'])
    # node_labs = dict(zip(np.arange(shp), tax_names_sort))
    sc = nx.draw_networkx_nodes(G=g, pos=pos, nodelist=g.nodes(),
            node_color=n_colors, node_size=node_szs,
            alpha=prps['node_alpha']
            )
    nx.draw_networkx_edges(G = g, pos = pos, width=edge_wts,
            edge_color=prps['edge_col'], alpha=prps['edge_alpha'])
    nx.draw_networkx_labels(G=g, pos=pos, labels=tax_names_dict,
            font_size=prps['ft'], verticalalignment=prps['v_algn'])

    output_bn = os.path.splitext(args.plot_adj_graph_fn)[0]
    ip.save_png_pdf(output_bn)

    return

if __name__ == '__main__':
    main()






#####
