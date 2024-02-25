import glob
import sys
import os
import gc
import argparse
import numpy as np
import aicspylibczi as aplc
import matplotlib.pyplot as plt
from collections import defaultdict




# =============================================================================


from cv2 import resize, INTER_CUBIC, INTER_NEAREST

def center_image(im, dims, ul_corner):
    shp = im.shape
    if not all([dims[i] == shp[i] for i in range(len(dims))]):
        shp_new = dims if len(shp) == 2 else dims + (shp[2],)
        temp = np.zeros(shp_new)
        br_corner = np.array(ul_corner) + np.array(shp[:2])
        temp[ul_corner[0]:br_corner[0], ul_corner[1]:br_corner[1]] = im
        im = temp
    return im

def resize_hipr(im, hipr_res, mega_res, dims='none', out_fn=False, ul_corner=(0,0)):
    # im = np.load(in_fn)
    factor_resize = hipr_res / mega_res
    hipr_resize = resize(
            im,
            None,
            fx = factor_resize,
            fy = factor_resize,
            interpolation = INTER_NEAREST
            )
    if isinstance(dims, str): dims = hipr_resize.shape
    hipr_resize = center_image(hipr_resize, dims, ul_corner)
    # if out_fn: np.save(out_fn, hipr_resize)
    return hipr_resize

def shift_mega(im):
    '''Globally define:  mega_shift_vector, max_shift, dims, ul_corner'''
    # im = np.load(in_fn)
    if len(im.shape) == 2: im = im[...,None]
    im = center_image(im, dims, ul_corner)
    return fsi._shift_images([im], mega_shift_vector, max_shift=max_shift)

def get_mega_props(seg, raw_shift, ch_):
    print(raw_shift.shape)
    raw = raw_shift[:,:,ch_]
    seg = seg.astype(np.int64)[:,:,0]
    return sf.measure_regionprops(seg, raw=raw)

def reshape_aics_image(m_img):
    '''
    Given an AICS image with just XY and CHannel,
    REshape into shape (X,Y,C)
    '''
    img = np.squeeze(m_img)
    img = np.transpose(img, (1,2,0))
    return img


# =============================================================================


def main():
    """
    Script to classify hiprfish images using a matrix based classifier.

    Usage:
        python sc_matrix_classify.py input_file output_file [--option1 OPTION1] [--option2 OPTION2]

    Example:
        python script_name.py data.csv processed_data.csv --option1 10 --option2 "value"
    """

    parser = argparse.ArgumentParser(description='Script Description')

    # Add arguments to the parser

    parser.add_argument('-cfn', '--config_fn', dest ='config_fn', 
                        type = str, help = 'Path to configureaiton file')
    parser.add_argument('-sn', '--sample_name', dest ='sn', 
                        type = str, help = 'Sample name wildcard')
    parser.add_argument('-od', '--out_dir', dest ='out_dir', 
                        type = str, help = 'Directory to save channel shifts.')  
    parser.add_argument('-m', '--m', dest ='m', 
                        type = str, help = 'z slice.') 
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Special imports 
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import image_plots as ip
    import fn_spectral_images as fsi

    # initialize direcotries
    for d in [args.out_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    sn = args.sample_name
    out_dir = args.out_dir
    im_inches = config['rgb_overlay']['im_inches']
    gauss = config['rgb_overlay']['gauss']
    spot_clims = config['rgb_overlay']['clims']
    clips = config['rgb_overlay']['clips']

    # Hiprfish Resolution 
    # Get raw data filename
    raw_fmt = config['data_dir'] + '/' + config['raw_fmt']
    raw_fn = raw_fmt.format(sample_name=sn, laser='488')
    # load  metadata
    # find resolution 
    czi = aplc.CziFile(raw_fn)
    for n in czi.meta.iter():
        if 'Scaling' in n.tag:
            if 'X' in n.tag:
                resolution = float(n.text)
    hipr_res_um_pix = resolution * 10**6
    # number of z stacks
    dims_shape = czi.get_dims_shape()[0]
    z_size = dims_shape['Z'][1]

    # Airyscan Resolution 
    # Get raw data filename
    raw_fmt = config['data_dir'] + '/' + sn + '_mode_airy_Airyscan Processing.czi'
    raw_mge_fn = raw_fmt.format(sample_name=sn, laser='488')
    # load  metadata
    # find resolution 
    czi_mge = aplc.CziFile(raw_mge_fn)
    for n in czi_mge.meta.iter():
        if 'Scaling' in n.tag:
            if 'X' in n.tag:
                resolution = float(n.text)
    mge_res_um_pix = resolution * 10**6
    # number of z stacks
    dims_shape = czi_mge.get_dims_shape()[0]
    z_size = dims_shape['Z'][1]

    # Load registered image
    shift_fmt = config['output_dir'] + '/' + config['shift_dir']
    af_str = re.sub('\.','_',str(af))
    shift_dir = shift_fmt.format(sample_name=sn, maskthresh=mt, allfluor=af_str)
    raw_fns = sorted(glob.glob(shift_dir + '/' + sn + '_M_' + str(m) + '_*'))
    raws = [np.load(f) for f in raw_fns]

    # pre process
    def get_smooth(m_raws_shift, sigma):
        raws_smooth = []
        for im in m_raws_shift:
            im_smooth = np.empty(im.shape)
            for i in range(im.shape[2]):
                im_smooth[:,:,i] = sf.pre_process(im[:,:,i], gauss=gauss)
                # im_smooth[:,:,i] = gaussian_filter(im[:,:,i], sigma=sigma)
            raws_smooth.append(im_smooth)
        return raws_smooth
    raws_smooth = get_smooth(raws, config['sigma'])
    stack_pre = np.dstack(raws_smooth)
    stack_pre_sum = np.sum(stack_pre, axis=2)
    stack_pre_sum_zoom = stack_pre_sum
    # ip.plot_image(np.sum(stack, axis=2),cmap='inferno', im_inches=im_inches)
    # plt.show()
    # plt.close()
    # ip.plot_image(stack_pre_sum_zoom, cmap='inferno', im_inches=im_inches)    
    # plt.show()
    # plt.close()

    # Plot RGB
    # clips = [(0.075,0.75),(0.075,0.3),(0.05,0.4)]
    rgb = [np.max(r, axis=2) for r in raws_smooth]
    rgb_smooth_adj_lst = []
    for r, clip in zip(rgb, clips):
        mx = np.max(rgb)
        mn = np.min(rgb)
        r_norm = (r - mn) / (mx - mn)
        r_adj = np.clip(r_norm, clip[0], clip[1])
        r_adj = (r_adj - clip[0]) / (clip[1] - clip[0])
        rgb_smooth_adj_lst.append(r_adj)
    rgb_smooth_adj = np.dstack(rgb_smooth_adj_lst)
    # for r in rgb_smooth_adj_lst:
        # ip.plot_image(r, cmap='inferno', im_inches=im_inches)
    # ip.plot_image(rgb_smooth_adj, im_inches=im_inches)
    # plt.show()
    # plt.close()

    # Get rescaled hiprfish image
    hipr_rgb_resize = resize_hipr(
            rgb_smooth_adj, hipr_res_um_pix, mge_res_um_pix
            )
    hipr_sum = np.sum(np.dstack(raws), axis=2)
    hipr_sum_resize = resize_hipr(
        hipr_sum, hipr_res_um_pix, mge_res_um_pix
        )
    
    # Load mge image 
    raw_mge, _ = czi_mge.read_image(Z=m)
    raw_mge = reshape_aics_image(raw_mge)
    mega_cell = raw_mge[:,:,mge_cell_chan]
    ip.plot_image(mega_cell, cmap='inferno')
    plt.show()
    plt.close

    # Which is the smaller image?
    mshp = mega_cell.shape[:2]
    hshp = hipr_sum_resize.shape[:2]
    im_list = [mega_cell, hipr_sum_resize]
    i_sml = np.argmin([mshp[0],hshp[0]])
    i_lrg = np.argmax([mshp[0],hshp[0]])
    sml = im_list[i_sml]
    lrg = im_list[i_lrg]
    # Get half the difference between sizes
    shp_dff = np.abs(np.array(hshp) - np.array(mshp)) // 2
    # Shift the smaller image so that it sits at the center of the larger image
    sml_shift_shape = lrg.shape[:2]
    if len(sml.shape) > 2:
        sml_shift += (sml.shape[2],)
    sml_shift = np.zeros(sml_shift_shape)
    corn_ind = np.array(shp_dff) + np.array(sml.shape[:2])
    sml_shift[shp_dff[0]:corn_ind[0], shp_dff[1]:corn_ind[1]] = sml
    # reassign mega and hipr image var names
    im_shift_list = [0,0]
    im_shift_list[i_sml] = sml_shift
    im_shift_list[i_lrg] = lrg
    mega_shift = im_shift_list[0]
    hipr_shift = im_shift_list[1]
    # Get the shift vectors for the mega image
    image_list = [hipr_shift, mega_shift]
    shift_vectors = fsi._get_shift_vectors(image_list)

    # Shift mge 
    max_shift = 500
    mega_shift_vector = [shift_vectors[1]]
    dims = lrg.shape
    ul_corner = shp_dff
    # run the shift function
    raw_shift = shift_mega(raw_mge)
    # cs_shifts = shift_mega(mega_cs)
    # ss_shifts = shift_mega(mega_ss)

    # show raw spot on top of classif
    ul = 0.15
    ll = 0.1
    spot_raw = raw_shift[0][:,:,1].copy()
    ip.plot_image(spot_raw, cmap='inferno')
    plt.show()
    plt.close()
    spot_raw = np.clip(spot_raw, spot_clims[0],spot_clims[1])
    spot_raw = (spot_raw - spot_clims[0]) / (spot_clims[1] - spot_clims[0])
    # spot_raw -= np.min(spot_raw)
    # spot_raw /= np.max(spot_raw)
    # spot_raw[spot_raw > ul] = ul
    # spot_raw[spot_raw < ll] = 0
    # spot_raw /= ul
    spot_raw_overlay = np.zeros(spot_raw.shape + (4,))
    spot_raw_overlay[:,:,0] = spot_raw
    spot_raw_overlay[:,:,2] = spot_raw
    spot_raw_overlay[:,:,3] = 1
    ip.plot_image(spot_raw_overlay)
    plt.show()
    plt.close()
    spot_raw_overlay[:,:,3] = spot_raw
    fig, ax, cbar = ip.plot_image(
            hipr_rgb_resize, scalebar_resolution=mge_res_um_pix, im_inches=im_inches
            )
    ax.imshow(spot_raw_overlay)
    out_bn = out_dir + '/' + sn + '_M_' + str(m)
    ip.save_png_pdf(out_bn)



if __name__ == "__main__":
    main()
