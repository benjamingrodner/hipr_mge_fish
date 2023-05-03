#################################################################################
# Packages
#################################################################################

import sys
sys.path.append('/fs/cbsuvlaminck2/workdir/bmg224/hiprfish/image_analysis_code')
from segmentation import Segmentation, javabridge
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
    parser.add_argument('-sprch', '--seg_plot_raw_channel', dest ='seg_plot_raw_channel', type = str, help = '')
    parser.add_argument('-clims', '--clims', dest ='clims', default=['min','max'], nargs='+', type = str, help = '')
    parser.add_argument('-sfm', '--seg_fname_mod', dest ='seg_fname_mod', default='_seg', type = str, help = '')
    parser.add_argument('-pfm', '--plot_fname_mod', dest ='plot_fname_mod', default='', type = str, help = '')
    parser.add_argument('-fff', '--flat_field_filename', dest ='flat_field_filename', default='', type = str, help = '')
    parser.add_argument('-ffi', '--flat_field_index', dest ='flat_field_index', default=0, type = int, help = '')

    args = parser.parse_args()
    
    im_seg = Segmentation(sample_name=args.sample_name, illumination_names=args.illumination_names, 
                          data_dir=args.data_dir, ext=args.ext, fname_mod=args.seg_fname_mod,
                          flat_field_filename=args.flat_field_filename,
                          flat_field_index=args.flat_field_index)                          
    im_seg.pick_channel(illumination=args.channel)
    if args.flat_field_filename:
        im_seg.flat_field_correction()
    im_seg.get_zoom_regions()
    im_seg.save_seg(raw_illumination=args.seg_plot_raw_channel, 
                    fname_mod=args.plot_fname_mod + '_zoom', zoom_coords=im_seg.zoom_coords,
                    clims=args.clims)
    im_seg.save_seg(raw_illumination=args.seg_plot_raw_channel, fname_mod=args.plot_fname_mod,
                    clims=args.clims)
        

if __name__ == '__main__':
    main()

javabridge.kill_vm()
