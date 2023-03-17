# Python script
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_10_03
# =============================================================================
"""
Given an input table of spectral czi,
register the shifts between laser lines in each sample
and save all the results as numpy files.

Configuration yaml file is required as input.
"""
# =============================================================================

import numpy as np
import  pandas as pd
import javabridge
import bioformats
import argparse
import yaml
import glob
import os
import sys
import re

javabridge.start_vm(class_path=bioformats.JARS)

# =============================================================================


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-f', '--fns', nargs='+', default=[])
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    parser.add_argument('-r', '--reg_fn', dest ='reg_fn', type = str, help = '')
    parser.add_argument('-m', '--max_fn', dest ='max_fn', type = str, help = '')
    parser.add_argument('-s', '--sum_fn', dest ='sum_fn', type = str, help = '')
    args = parser.parse_args()

    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi

    # Load image table
    # im_tab_fn = config['images']['image_list_table']
    # im_tab = pd.read_csv(im_tab_fn)

    # print('Preparing', str(im_tab.shape[0]), 'samples')

    # Load images
    out_dir = (config['output_dir'] + '/' + config['prep']['out_dir'])
    # if not os.path.exists(out_dir): os.makedirs(out_dir)
    # print(im_tab['IMAGES'].tolist())
    # for sn in im_tab['IMAGES'].tolist():
    #     bn = config['__default__']['DATA_DIR'] + '/' + sn
    #     fns = glob.glob(bn + config['laser_regex'])
    image_list = []
    for fn in args.fns:
        out_fn = out_dir + '/' + os.path.split(os.path.splitext(fn)[0])[1] + '.npy'
        if not os.path.exists(out_fn):
            image = bioformats.load_image(fn)

            # save separate lasers to numpy
            lookahead = config['prep']['laser_lookahead']
            laser = re.search(r'\d{3}(?=' + lookahead + ')',fn).group(0)
            np.save(out_fn, image)
        else:
            image = np.load(out_fn)
        image_list.append(image)

    # Register shifts
    im_reg, im_max, im_sum, shift_vectors = fsi.register_shifts(
            image_list,
            config['max_shift']
            )

    # ## TEST
    # # No image registration
    # im_reg = np.dstack(image_list)
    # im_max = np.max(im_reg, axis=2)
    # im_sum = np.sum(im_reg, axis=2)
    # # shift_vectors =
    # ## TEST

    # Save registered image
    # out_fn = (
    #         config['output_dir'] + '/'
    #         + config['reg_fmt'].format(sample_name=sn)
    #         )
    np.save(args.reg_fn, im_reg)

    # Save max projection
    # out_fn = (
    #         config['output_dir'] + '/'
    #         + config['max_fmt'].format(sample_name=sn)
    #         )
    np.save(args.max_fn, im_max)

    # save sum projection
    # out_fn = (
    #         config['output_dir'] + '/'
    #         + config['sum_fmt'].format(sample_name=sn)
    #         )
    np.save(args.sum_fn, im_sum)
    # print('Prepped files for sample:', sn)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
