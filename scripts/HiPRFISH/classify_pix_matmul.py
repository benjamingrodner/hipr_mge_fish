
"""
Ben Grodner 2023
De Vlaminck Lab
Cornell University

HiPRFISH processing pipeline script

Classify pixels in registered image using matrix multiplication method
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def main():
    parser = argparse.ArgumentParser('Plot the classification results')
    parser.add_argument('-c', '--config_fn', dest ='config_fn', type = str, help = '')
    # parser.add_argument('-sp', '--seg_props', dest = 'seg_props', type = str, default = '', help = 'Input normalized single cell spectra filename')
    parser.add_argument('-rf', '--reg_fn', dest = 'reg_fn', type = str, default = '', help = 'Input classified spectra table')
    parser.add_argument('-of', '--out_fn', dest = 'out_fn', type = str, default = '', help = 'Input classified spectra table')

    # parser.add_argument('-r', '--ref_clf', dest = 'ref_clf', type = str, default = '', help = 'Spectra classifier path')
    args = parser.parse_args()

    # set  parameters in config file
    with open(args.config_fn, 'r') as f:
        config = yaml.safe_load(f)

    # Load specialized modules
    sys.path.append(config['pipeline_path'] + '/' + config['functions_path'])
    import fn_spectral_images as fsi
    import image_plots as ip
    import segmentation_func as sf

    stack = np.load(args.reg_fn)

    # smoothing
    sigma=config['chan_gauss']

    stack_smooth = np.empty(stack.shape)
    for i in range(stack.shape[2]):
        stack_smooth[:,:,i] = gaussian_filter(stack[:,:,i], sigma=sigma)

    # Get maximum intensity mask
    pdict = config['cell_seg']
    im = sf.max_projection(stack, pdict['channels'])
    mask = sf.get_background_mask(
        im,
        bg_filter=pdict['bg_filter'],
        bg_log=pdict['bg_log'],
        bg_smoothing=pdict['bg_smoothing'],
        n_clust_bg=pdict['n_clust_bg'],
        top_n_clust_bg=pdict['top_n_clust_bg'],
        bg_threshold=pdict['bg_threshold'],
        bg_file=bg_file
        )

    np.save(args.mask_fn, mask)

    max_unsmooth = np.max(stack, axis=2)
    # sum_sub = np.sum(stack, axis=2)
    max_sub = np.max(stack_smooth, axis=2)
    # mask = max_sub > max_thresh
    max_masked = max_sub * mask
    ip.subplot_square_images([max_unsmooth, mask, max_masked], (1,3))
    out_bn = os.path.splitext(args.plot_mask_fn)
    ip.save_png_pdf(out_bn)

    # Scale pixel spectra
    spec_pix_norm = spec_pix / np.max(spec_pix, axis=1)[:,None]
    pix_scaler = StandardScaler().fit(spec_pix)
    spec_pix_scaled = pix_scaler.transform(spec_pix)


    probe_design_fn = probe_design_dir + '/' + config['probe_design_filename']
    df_probe_design = pd.read_csv(probe_design_fn)
    barcodes = df_probe_design['code'].values
    barcodes = df_probe_design['sci_name'].values
    dict_bc_tax = dict(zip(barcodes, sci_names))

    # max normalize
    ref_spec = []
    for bc in barcodes:
        fn = ref_dir + '/'+ fmt.format(bc)
        ref = pd.read_csv(fn)[avgint_cols].values
        ref_spec.append(ref)
    # ref_norm = ref / np.max(ref, axis=1)[:,None]
    # weights.append(np.mean(ref_norm, axis=0))

# %% codecell
# max normalized
ref_norm = [r / np.max(r, axis=1)[:,None] for r in ref_spec]
weights_max_norm = [np.mean(r, axis=0) for r in ref_norm]

# %% codecell
# sum normalized
ref_sum_norm = []
for r in ref_spec:
    r_ = r - np.min(r, axis=1)[:,None]
    ref_sum_norm.append(r_ / np.sum(r_, axis=1)[:,None])
weights_sum_norm = [np.mean(r, axis=0) for r in ref_sum_norm]
len(weights_sum_norm)

# %% codecell
# Only maximum values used in evaluation
top_n = 5
weights_top_n = []
for w in weights_max_norm:
    l = len(w)
    ind = np.argpartition(w, l-top_n)[:l-top_n]
    w_ = w.copy()
    w_[ind] = 0
    weights_top_n.append(w_)

# %% codecell
# show weights
weights_pick = weights_sum_norm

for w, n in zip(weights_pick, sci_names):
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, np.array(w)[None,:], {'lw':1,'alpha':1,'color':'r'})
    ax.set_title(n)

# %% codecell
input = spec_pix_scaled

weights_t = np.array(weights_pick).T
classif_mat = np.matmul(input, weights_t)
classifs_index = np.argmax(classif_mat, axis=1)
classifs = np.array([sci_names[i] for i in classifs_index])

# %% codecell
n = 1000
labs = np.unique(classifs, return_counts=True)
labs_sort = [x for _, x in sorted(zip(labs[1],labs[0]), reverse=True)]
for l in labs_sort[:20]:
    bool = (classifs == l)
    spec_ = spec_pix_norm[bool,:]
    inds = np.random.randint(0,spec_.shape[0], n)
    spec_rand = spec_[inds,:]
    fig, ax = ip.general_plot(dims=(10,5))
    fsi.plot_cell_spectra(ax, spec_rand, {'lw':0.1,'alpha':0.1,'color':'r'})
    ax.set_title(l)

# %% codecell
# Re project the pixels onto the image
# Get the indices for each pixeel after masking
pix_ind = np.argwhere(mask)
# for each index draw the color
probe_design_dir = config['__default__']['PROBE_DESIGN_DIR']
df_bc_col = pd.read_csv(probe_design_dir + '/' + config['barcode_color_fn'])
colors = df_bc_col['color'].apply(eval)
col_dict = dict(zip(df_bc_col['barcode'], colors))
# tab10 = plt.get_cmap('tab10').colors
# col_dict = {l:c for l, c in zip(labs_sort, tab10)}
im_clust = np.zeros(max_sub.shape + (len(tab10[0]),))
for lab, i in zip(classifs, pix_ind):
    x, y = i
    col = col_dict[lab]
    # col = np.array(col_dict[lab]) * sum_sub_norm[x,y]
    im_clust[x,y,:] = np.array(col) * max_unsmooth[x,y]

# %% codecell
im_inches=10
ip.plot_image(im_clust, im_inches=im_inches)
# output_dir = config['output_dir'] + '/test_classif/mat_mul'
# if not os.path.exists(output_dir): os.makedirs(output_dir)
# out_bn = out_dir + '/' + sn + '_zoom_classif_col'
# ip.save_png_pdf(out_bn)
os.getcwd()

# %% codecell
col_ordered = [col_dict[l] for l in labs_sort]
fig, ax = ip.taxon_legend(
        taxon_names=labs_sort,
        taxon_colors=col_ordered
        )


    return

if __name__ == '__main__':
    main()






#####
