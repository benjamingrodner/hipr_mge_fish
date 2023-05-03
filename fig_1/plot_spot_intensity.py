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
# Params
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
    parser.add_argument('-pl', '--plots', dest ='plots', nargs='+', default=['False'], type=str, help='')
    parser.add_argument('-pf', '--plot_factor', dest ='plot_factor', default='', type=str, help='')
    parser.add_argument('-gr', '--groups', dest ='groups', nargs='+', type=str, help='')
    parser.add_argument('-gf', '--group_factor', dest ='group_factor', type=str, help='')
    parser.add_argument('-gl', '--group_labels', dest ='group_labels', nargs='+', type=str, help='')
    parser.add_argument('-sgr', '--subgroups', dest ='subgroups', nargs='+', default='', type=str, help='')
    parser.add_argument('-sgf', '--subgroup_factor', dest ='subgroup_factor', default='', type=str, help='')
    parser.add_argument('-sgl', '--subgroup_labels', dest ='subgroup_labels', nargs='+', default=[''], type=str, help='')
    parser.add_argument('-sgs', '--subgroup_spacer', dest ='subgroup_spacer', default=0.1, type=float, help='')
    parser.add_argument('-xl', '--xlabel', dest ='xlabel', default='', type=str, help='')
    parser.add_argument('-yl', '--ylabel', dest ='ylabel', default='Spot Intensity (a.u.)', type=str, help='')
    parser.add_argument('-ft', '--ft', dest ='ft', default=12, type=int, help='')
    parser.add_argument('-dims', '--dims', dest='dims', nargs='+', default=(5,3), type=float, help='')
    parser.add_argument('-lims', '--lims', dest='lims', nargs='+', default=('min','max'), type=str, help='')
    parser.add_argument('-col', '--color', dest='color', default='k', type=str, help='')
    parser.add_argument('-jt', '--jit', dest='jit', default=0.1, type=float, help='')
    parser.add_argument('-df', '--dot_factor', dest='dot_factor', default=0.5, type=float, help='')
    parser.add_argument('-tr', '--transparency', dest='transparency', default=0.7, type=float, help='')
    parser.add_argument('-ls', '--log_scale', dest='log_scale', default=1, type=int, help='')
    args = parser.parse_args()
    
#################################################################################
# Script
#################################################################################

    # Get params and format properly
    spot_seg = [Segmentation(sample_name=sn, 
                             data_dir=args.data_dir, ext=args.raw_ext,
                             illumination_names=args.illumination_names, 
                             fname_mod=args.seg_fname_mod_spot)
                for sn in args.sample_names]
    keys = [seg.get_filename_keys(args.factors) for seg in spot_seg]
    key_seg_zip = zip(keys, spot_seg)        
    plots = [False] if args.plots[0] == 'False' else args.plots
    subgroups = [False] if args.subgroups[0] == 'False' else args.subgroups    
    pfac_index = args.factors.index(args.plot_factor) if plots[0] else np.nan
    gfac_index = args.factors.index(args.group_factor)
    sgfac_index = args.factors.index(args.subgroup_factor) if subgroups[0] else np.nan
    
    # Set up plot axes
    for p in plots:
        fig, ax = ip.general_plot(xlabel=args.xlabel, ylabel=args.ylabel, ft=args.ft, dims=args.dims,
                                  col=args.color)
        xticks = ip.get_xticks(args.groups, subgroups, args.subgroup_spacer)
        xticklabels = ip.get_xtick_sublabels(xticks, args.group_labels, args.subgroup_labels)
        ax.set_xticklabels(xticklabels, fontsize=args.ft*4//5)
        ax.set_xticks(xticks)
        ax.set_xlim(np.min(xticks) - args.subgroup_spacer, np.max(xticks) + args.subgroup_spacer)
        xticks_removed = ip.remove_major_ticks(xticks, args.groups, args.subgroups)
        
        # Gather intensity values
        key_seg_plot = [ks for ks in key_seg_zip if p == ks[0][pfac_index]] if plots[0] else list(key_seg_zip)
        values = []
        positions = []
        i = 0
        for g in args.groups:
            key_seg_group = [ksp for ksp in key_seg_plot if g == ksp[0][gfac_index]]
            for sg in subgroups:
                key_seg_sg = [ks for ks in key_seg_group if sg == ks[0][sgfac_index]] \
                                if subgroups[0] else key_seg_group
                for seg in key_seg_sg:
                    values_filename = seg[1].region_props_filename
                    values_ = pd.read_csv(values_filename).Intensity.values
                    values_ = np.log10(values_ + 1e-15) if args.log_scale else values_
                    values.append(values_)
                    positions.append(xticks_removed[i])
                i += 1
                    
        # Plot intensity values
        ip.violin_dot_plot(ax, values, positions=positions, jit=args.jit, ft=args.ft, 
                           dot_factor=args.dot_factor, transparency=args.transparency,
                           col=args.color)
        ip.adjust_ylims(ax, values, lims=args.lims, log_scale=args.log_scale, ft=args.ft, ylabel=args.ylabel)
        
        # Save the plot
        plot_append = '_' + p if plots[0] else ''
        fig_basename = args.fig_basename + plot_append
        plot_filename = args.fig_dir + '/' + fig_basename + args.fig_ext
        ip.plt.savefig(plot_filename, transparent=True)
        print('Wrote: ', plot_filename)

    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
