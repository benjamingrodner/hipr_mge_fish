#################################################################################
# Packages
#################################################################################

import sys
sys.path.append('/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code')
from object_segmentation import Segmentation, javabridge
import image_functions as imfn
import argparse

#################################################################################
# Script execution
#################################################################################


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('sample_name', type = str, help = 'Input sample base name')
    parser.add_argument('-ilnm', '--illumination_names', dest = 'illumination_names', nargs='+', type = str, help = '')
    parser.add_argument('-dd', '--data_dir', dest ='data_dir', type = str, help = '')
    parser.add_argument('-ext', '--ext', dest ='ext', type = str, help = '')
    parser.add_argument('-ch', '--channel', dest ='channel', type = str, help = '')
    parser.add_argument('-bfc', '--bg_filter_channel', dest ='bg_filter_channel', default='', type = str, help = '')
    parser.add_argument('-sfm', '--seg_fname_mod', dest ='seg_fname_mod', type = str, help = '')
    parser.add_argument('-lsm', '--lne_smoothing', dest ='lne_smoothing', default=0, type = int, help = '')
    parser.add_argument('-bgl', '--bg_log', dest ='bg_log', default=0, type = int, help = '')
    parser.add_argument('-lnel', '--lne_log', dest ='lne_log', default=0, type = int, help = '')
    parser.add_argument('-fff', '--flat_field_filename', dest ='flat_field_filename', default='', type = str, help = '')
    parser.add_argument('-ffi', '--flat_field_index', dest ='flat_field_index', default=0, type = int, help = '')
    parser.add_argument('-ncl', '--n_clusters', dest ='n_clusters', default=2, type = int, help = '')
    parser.add_argument('-tpn', '--top_n_clusters', dest ='top_n_clusters', default=1, type = int, help = '')
    parser.add_argument('-bgf', '--bg_filter', dest ='bg_filter', default=1, type = int, help = '')
    parser.add_argument('-lwm', '--lne_watershed_mask', dest ='lne_watershed_mask', default=0, type = int, help = '')
    args = parser.parse_args()
    
    im_seg = Segmentation(sample_name=args.sample_name, 
                          illumination_names=args.illumination_names, 
                          data_dir=args.data_dir, ext=args.ext, fname_mod=args.seg_fname_mod,
                          lne_smoothing=args.lne_smoothing, lne_log=args.lne_log,
                          bg_log=args.bg_log, n_clusters=args.n_clusters,
                          top_n_clust=args.top_n_clusters, bg_filter=args.bg_filter,
                          bg_filter_channel=args.bg_filter_channel,
                          flat_field_filename=args.flat_field_filename,
                          flat_field_index=args.flat_field_index)
    im_seg.pick_channel(illumination=args.channel)
    if args.flat_field_filename:
        print('Correcting flat field...')
        im_seg.flat_field_correction()
    im_seg.segment(lne_watershed_mask=args.lne_watershed_mask)
#     im_seg.get_zoom_regions()
#     im_seg.save_seg_process(zoom_coords=im_seg.zoom_coords)
#     im_seg.measure_regionprops()
        

if __name__ == '__main__':
    main()

javabridge.kill_vm()
