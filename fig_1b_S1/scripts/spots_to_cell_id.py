#################################################################################
# Packages
#################################################################################

import sys
sys.path.append('/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code')
# from segmentation import Segmentation, javabridge
import image_functions as imfn
import argparse
import numpy as np
import pandas as pd
   
#################################################################################
# Script execution
#################################################################################

def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('sample_name', type = str, help = 'Input sample base name')
    parser.add_argument('-ilnm', '--illumination_names', dest = 'illumination_names', nargs='+', type = str, help = '')
    parser.add_argument('-dd', '--data_dir', dest ='data_dir', type = str, help = '')
    parser.add_argument('-od', '--out_dir', dest ='out_dir', type = str, help = '')
    parser.add_argument('-ext', '--ext', dest ='ext', type = str, help = '')
    parser.add_argument('-sfmc', '--seg_fname_mod_cell', dest ='seg_fname_mod_cell', default='_cell_seg', type = str, help = '')
    parser.add_argument('-sfms', '--seg_fname_mod_spot', dest ='seg_fname_mod_spot', default='_spot_seg', type = str, help = '')
    parser.add_argument('-sr', '--search_radius', dest ='search_radius', default=10, type = int, help = '')
    parser.add_argument('-ds', '--downsample', dest ='downsample', default=1, type = int, help = '')
    args = parser.parse_args()

#     mods = [args.seg_fname_mod_cell, args.seg_fname_mod_spot]
#     segs = [Segmentation(sample_name=args.sample_name, data_dir=args.data_dir, ext=args.ext,
#                             illumination_names=args.illumination_names, 
#                             fname_mod=fm) for fm in mods]
#     cell_seg_array = np.load(segs[0].seg_filename)
#     spot_props = pd.read_csv(segs[1].region_props_filename)
#     spot_props_cell_id = imfn.spots_to_cell_id(cell_seg_array, spot_props, 
#                                                args.search_radius, args.downsample)
#     output_filename = args.out_dir + '/' + args.sample_name +\
#                       args.seg_fname_mod_spot + '_cell_id.csv'
#     spot_props_cell_id.to_csv(output_filename)


#     sn=args.sample_name
#     downsample=2
#     search_radius=20
#     image_processing_dir='../image_processing'
#     # for each sample name
#     spot_props = pd.DataFrame()
#     cell_props = pd.DataFrame()
#     spot_split_folder = image_processing_dir + '/' + sn + '_spot_seg_split'
#     cell_split_folder = image_processing_dir + '/' + sn + '_cell_seg_split'
#     spot_split_fns = imfn.get_sample_names(data_dir=spot_split_folder, sample_glob='*col_[012]', ext='.npy')
#     cell_split_fns = imfn.get_sample_names(data_dir=cell_split_folder, sample_glob='*col_[012]', ext='.npy')
#     spot_split_fns = [spot_split_folder + '/' + i for i in spot_split_fns]
#     cell_split_fns = [cell_split_folder + '/' + i for i in cell_split_fns]
#     # for each split file
#     for sfn, cfn in zip(spot_split_fns, cell_split_fns):
#         ss_fn = sfn + '_seg.npy'
#         sr_fn = sfn + '.npy'
#         cs_fn = cfn + '_seg.npy'
#         cr_fn = cfn + '.npy'
#         # Load cell and spot numpy segs and raw
#         print('Loading files...')
#         ss = np.load(ss_fn)
#         cs = np.load(cs_fn)
#         sr = np.load(sr_fn)
#         cr = np.load(cr_fn)
#         print('Loaded files')
#         # Get spot and cell object props
#         print('measuring regionprops...')
#         sp = imfn.measure_regionprops(ss, sr)
#         cp = imfn.measure_regionprops(cs, cr)
#         print('Measured regionprops')
#         # Get spot cell ids
#         print('Identifying parent cells...')
#         sp = imfn.spots_to_cell_id(cs, 
#                                    sp,
#                                    search_radius, 
#                                    downsample)
#         print('Identified parent cells')
#         # add unique identifier to object id
# #         print(spot_props.shape[0])
# #         sp_max_label = spot_props.ID.max() if spot_props.shape[0] > 0 else 0
# #         cp_max_label = cell_props.ID.max() if cell_props.shape[0] > 0 else 0
# #         sp['id_unique'] = sp.ID + sp_max_label
# #         cp['id_unique'] = cp.ID + cp_max_label
# #         # merge props with sample name csv    
# #         spot_props = spot_props.append(sp, ignore_index=True)        
# #         cell_props = cell_props.append(cp, ignore_index=True)
# #         output_sp_fn = sfn + '_spot_seg_cell_id.csv'
# #         output_cp_fn = cfn + '_cell_seg_props.csv'
# #         spot_props.to_csv(output_sp_fn)
# #         cell_props.to_csv(output_cp_fn)
#         output_sp_fn = sfn + '_spot_seg_cell_id.csv'
#         output_cp_fn = cfn + '_cell_seg_props.csv'
#         sp.to_csv(output_sp_fn)
#         cp.to_csv(output_cp_fn)  
    out_dir = '../image_processing'
    # Load split dirs
    spot_split_dirs = out_dir + '/' + args.sample_name + '_spot_seg_split' 
    cell_split_dirs = out_dir + '/' + args.sample_name + '_cell_seg_split'
    # Get split names in specific order
    import glob
    import re
    import os
    spot_segs_split = glob.glob(spot_split_dirs + '/*seg.npy')
    split_names = [re.sub('_seg.npy','',os.path.split(n)[1]) for n in spot_segs_split]  
    # load raws and segs
#     spot_raws_split = [np.load(spot_split_dirs + '/' +  n + '.npy') for n in split_names]
#     spot_segs_split = [np.load(spot_split_dirs + '/' +  n + '_seg.npy') for n in split_names]
#     spot_props_split = [pd.read_csv(spot_split_dirs + '/' +  n + '_props.csv') for n in split_names]
    cell_raws_split = [np.load(cell_split_dirs + '/' +  n + '.npy') for n in split_names]
    cell_segs_split = [np.load(cell_split_dirs + '/' +  n + '_seg.npy') for n in split_names]
    search_radius = 20
    downsample = 2
#     spot_segs_split_filt = [s * (c > 0) for s, c in zip(spot_segs_split, cell_segs_split)]
#     for cs, sp,  n in zip(cell_segs_split, spot_props_split, split_names):
#         out_filename = spot_split_dirs + '/' + n + '_cell_id.csv'
#         props = imfn.spots_to_cell_id(cs, 
#                            sp,
#                            search_radius, 
#                            downsample)
#         props.to_csv(out_filename)
#         print('Wrote: ', out_filename)
        
    for cs, cr,  n in zip(cell_segs_split, cell_raws_split, split_names):
        out_filename = cell_split_dirs + '/' + n + '_props.csv'
        props = imfn.measure_regionprops(cs,cr)
        props.to_csv(out_filename)
        print('Wrote: ', out_filename)

    pd.DataFrame([]).to_csv(out_dir + '/' + args.sample_name + '_spot_seg_cell_id.csv')
    return

if __name__ == '__main__':
    main()

# javabridge.kill_vm()
