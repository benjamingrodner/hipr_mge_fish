#################################################################################
# Packages
#################################################################################

import sys
sys.path.append('/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code')
from segmentation import Segmentation, javabridge
import image_plots as ip
import image_functions as imfn
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import ks_2samp

#################################################################################
# Script execution
#################################################################################

def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('sample_names', nargs='+', type = str, help = 'Input sample base name')
    parser.add_argument('-fac', '--factors', dest='factors', nargs='+', type=str, help='')
    parser.add_argument('-ilnm', '--illumination_names', dest='illumination_names', nargs='+', type=str, help='')
    parser.add_argument('-dd', '--data_dir', dest='data_dir', type=str, help='')
    parser.add_argument('-rext', '--raw_ext', dest='raw_ext', type=str, help='')
    parser.add_argument('-fdir', '--fig_dir', dest='fig_dir', type=str, help='')
    parser.add_argument('-fbn', '--fig_basename', dest='fig_basename', default='ks_test', type=str, help='')
    parser.add_argument('-fext', '--fig_ext', dest ='fig_ext', default='.png', type = str, help = '')
    parser.add_argument('-sfms', '--seg_fname_mod_spot', dest ='seg_fname_mod_spot', default='_spot_seg', type=str, help='')
    parser.add_argument('-pl', '--plots', dest ='plots', nargs='+', type=str, help='')
    parser.add_argument('-gr', '--groups', dest ='groups', nargs='+', type=str, help='')
    parser.add_argument('-gl', '--group_labels', dest ='group_labels', nargs='+', type=str, help='')
    parser.add_argument('-pf', '--plot_factor', dest ='plot_factor', default='', type=str, help='')
    parser.add_argument('-gf', '--group_factor', dest ='group_factor', type=str, help='')
    parser.add_argument('-sgf', '--subgroup_factor', dest ='subgroup_factor', default='', type=str, help='')
    parser.add_argument('-ctf', '--control_factor', dest ='control_factor', type=str, help='')
    parser.add_argument('-tf', '--test_factor', dest ='test_factor', type=str, help='')
    parser.add_argument('-xl', '--xlabel', dest ='xlabel', default='Infection time (min)', type=str, help='')
    parser.add_argument('-yl', '--ylabel', dest ='ylabel', default='Kolmogorov-Smirnov Statistic', type=str, help='')
    parser.add_argument('-ft', '--ft', dest ='ft', default=12, type=int, help='')
    parser.add_argument('-dims', '--dims', dest='dims', nargs='+', default=(5,3), type=float, help='')
    parser.add_argument('-lims', '--lims', dest='lims', nargs='+', default=(0,1), type=int, help='')
    parser.add_argument('-col', '--color', dest='color', default='k', type=str, help='')
    args = parser.parse_args()
    
    ###

    # Gather parameters and format properly
    spot_seg = [Segmentation(sample_name=sn, 
                             data_dir=args.data_dir, ext=args.raw_ext,
                             illumination_names=args.illumination_names, 
                             fname_mod=args.seg_fname_mod_spot)
                for sn in args.sample_names]
    keys = [seg.get_filename_keys(args.factors) for seg in spot_seg]
    key_seg_zip = zip(keys, spot_seg) 
    plots = [False] if args.plots[0] == 'False' else args.plots
    pfac_index = args.factors.index(args.plot_factor) if plots[0] else np.nan
    gfac_index = args.factors.index(args.group_factor)
    sgfac_index = args.factors.index(args.subgroup_factor)
    
    # Set up plot axes
    for p in plots:
        fig, ax = ip.general_plot(xlabel=args.xlabel, ylabel=args.ylabel, ft=args.ft, dims=args.dims,
                                  col=args.color)
        xticks = ip.get_xticks(args.groups)
        xticklabels = args.group_labels
        ax.set_xticklabels(xticklabels, fontsize=args.ft*4//5)
        ax.set_xticks(xticks)
        
        # Gather ks stats and plot
        key_seg_plot = [ks for ks in key_seg_zip if p == ks[0][pfac_index]] if plots[0] else list(key_seg_zip)
        D_list = []
        for g in args.groups:
            key_seg_group = [ksp for ksp in key_seg_plot if g == ksp[0][gfac_index]]
            key_seg_controls = [ksg for ksg in key_seg_group if args.control_factor == ksg[0][sgfac_index]]
            key_seg_test = [ksg for ksg in key_seg_group if args.test_factor == ksg[0][sgfac_index]]
            D_g = []
            for c in key_seg_controls:
                for t in key_seg_test:
                    filenames = [j[1].region_props_filename for j in [c,t]]
                    props = [pd.read_csv(f) for f in filenames]
                    values = [p.Intensity for p in props]
                    D, P = ks_2samp(values[1], values[0])
                    D_g.append(D)
            D_list.append(D_g)
        ip.bar_plot(ax, D_list, xticks, ft=args.ft, col=args.color)
        ip.adjust_ylims(ax, D_list, args.lims)
        
        # Save plot
        plot_append = '_' + p if plots[0] else ''
        fig_basename = args.fig_basename + plot_append
        plot_filename = args.fig_dir + '/' + fig_basename + args.fig_ext
        ip.plt.savefig(plot_filename, transparent=True)
        print('Wrote: ', plot_filename)

    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
