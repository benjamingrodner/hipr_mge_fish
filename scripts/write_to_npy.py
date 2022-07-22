# Python script
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_28
# =============================================================================
"""
Write a file to npy using bioformats load
"""
# =============================================================================

import numpy as np
import javabridge
import bioformats
import argparse

javabridge.start_vm(class_path=bioformats.JARS)

# =============================================================================


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('in_fn', type = str, help = 'Input filename')
    parser.add_argument('out_fn', type = str, help = 'Output filename')
    args = parser.parse_args()

    image = bioformats.load_image(args.in_fn)
    np.save(args.out_fn, image)
    print('Wrote: ', args.out_fn)
    return

if __name__ == '__main__':
    main()

javabridge.kill_vm()
