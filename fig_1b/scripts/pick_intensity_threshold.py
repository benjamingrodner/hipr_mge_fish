# %% codecell
# Purpose: Show normalized threshold curves to pick lower intensity
    # threshold
# %% codecell
%load_ext autoreload
%autoreload 2

# %% codecell
cluster_path = '/fs/cbsuvlaminck2/'
import sys
sys.path.append(cluster_path + 'workdir/bmg224/hiprfish/image_analysis_code')
import image_functions as imfn

data_dir = cluster_path + '/workdir/bmg224/data/2021/brc_fileshare_new/2021_06_02_plasmidairyscan/processed'
factors = ['exp','method', 'plasmid', 'fov']
out_dir = '../image_processing'
sample_names = imfn.get_sample_names(data_dir=data_dir)
print(len(sample_names))
refl = ['gfp', 'cy5']
# %% codecell
keys = [imfn.get_filename_keys(sn, filename_factors=factors) for sn in sample_names]
key_labels = [", ".join(k) for k in keys]
print(len(keys))
# %% codecell

# %% codecell
sn_dict = imfn.get_nested_dict(keys, sample_names, [0,1,2])
H = ['040421', '080220']
I = ['a','b','c','d','e','f','g','h']
J = ['neg','pos']
K = ['1','2','3']
sn_sort = []
key_labels = []
for i in I:
    for j in J:
        for k in K:
            for key, sn in sn_dict[i][j][k]:
                sn_sort.append(sn)
                key_labels.append(', '.join(key))
# %% codecell
# Get spot properties
import pandas as pd
spot_props_filenames = ['{}/{}_spot_seg_cell_id.csv'.format(out_dir, sn) for sn in sample_names]
spot_props = [pd.read_csv(spf) for spf in spot_props_filenames]
# %% codecell
# Filter by distance
sp_dist = [sp.loc[sp.dist <= 10, :] for sp in spot_props]
# %% codecell
# build dictionary
values = list(zip(sample_names, sp_dist))
sp_dict = imfn.get_nested_dict(keys, values, [0,1,2,3])
# %% codecell
# Plot standard histograms

import image_plots as ip
import numpy as np

fig_dir = '../figures'

col = 'w'
lw = 4
# lw = 1
ft = 20
# ft = 5
dims = (10,5)

I = ['a','b','c','d','e','f','g','h']
J = ['neg','pos']
K = ['1','2','3']
# dims = (0.9843 ,0.5906)
# xlabel = 'Spot Intensity'
# ylabel = 'Frequency'
# Produce a plot for each method
for m in I:
    for method, meth_dict in sp_dict.items():
        if method == m:
            print('method: ', method)
        #     if method == 'e':
        #         fig, ax = ip.general_plot(xlabel=xlabel,ylabel=ylabel, col=col, ft=ft)
        #     else:
            fig, ax = ip.general_plot(col=col, dims=dims, lw=lw)
            # Plot one line for each control
            line_styles = [':','-']
            legend_labels = ['Negative','Postitive']
            plasmid = ['neg','pos']
            for p in J:
                # Combine the values for reps and fovs for cells
                value_list = []
#             for p, ls, ll in zip(plasmid, line_styles, legend_labels):
                for k in K:
                    tup_list = meth_dict[p][k]

                    for keys, values in tup_list:
                        # Exclude overexposed images
            #             if method in ['c','d']:
            #                 if keys[3] in ['1','2','3']:
            #                     pass
            #                 elif keys[3][0] != 'b':
            #                     value_list += values.tolist()
            #             else:
            #                 # Exclude bg images
            #                 if keys[3][0] != 'b':
                        print(values[0])
                        v = values[1].Intensity.values.tolist()
    #                     value_list += values[1].Intensity.values.tolist()

                        y, bin_edges = np.histogram(v, bins=100)
                        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    #                     y_norm = y/np.max(y)
                        # Log scale
    #                     bin_log = np.log10(bin_centers + 1e-15)
    #                     ax.plot(bin_log, y_norm, label=values[0][31:])
    #                     ax.plot(bin_centers, y_norm, label=values[0][31:])
                        ax.plot(bin_centers, y, label=values[0][31:])
#                 ax.plot(bin_centers, y_norm, linestyle=ls, lw=lw, label=ll)
#             if method == 'a':
            ax.legend(fontsize=ft)
#             ax.set_xlim((0.15,0.4))
        #     ax.set_ylim((0,4500000))
        #     for side in ['top','bottom','left','right']:
        #         ax.spines[side].set_linewidth(4)
    #         [side.set_linewidth(4) for side in ax.spines.values()]
#             ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            filename = fig_dir + '/spot_int_histogram_method_' + method + '.pdf'
            ip.plt.show()
#             ip.plt.savefig(filename, transparent=True)
            ip.plt.close()

# %% codecell
# Plot log histograms

import image_plots as ip
import numpy as np

fig_dir = '../figures'

col = 'w'
lw = 4
# lw = 1
ft = 20
# ft = 5
dims = (10,5)

I = ['a','b','c','d','e','f','g','h']
J = ['neg','pos']
K = ['1','2','3']
# dims = (0.9843 ,0.5906)
# xlabel = 'Spot Intensity'
# ylabel = 'Frequency'
# Produce a plot for each method
for m in I:
    for method, meth_dict in sp_dict.items():
        if method == m:
            print('method: ', method)
        #     if method == 'e':
        #         fig, ax = ip.general_plot(xlabel=xlabel,ylabel=ylabel, col=col, ft=ft)
        #     else:
            fig, ax = ip.general_plot(col=col, dims=dims, lw=lw)
            # Plot one line for each control
            line_styles = [':','-']
            legend_labels = ['Negative','Postitive']
            plasmid = ['neg','pos']
            for p in J:
                # Combine the values for reps and fovs for cells
                value_list = []
#             for p, ls, ll in zip(plasmid, line_styles, legend_labels):
                for k in K:
                    tup_list = meth_dict[p][k]

                    for keys, values in tup_list:
                        # Exclude overexposed images
            #             if method in ['c','d']:
            #                 if keys[3] in ['1','2','3']:
            #                     pass
            #                 elif keys[3][0] != 'b':
            #                     value_list += values.tolist()
            #             else:
            #                 # Exclude bg images
            #                 if keys[3][0] != 'b':
                        print(values[0])
                        v = values[1].Intensity.values.tolist()
    #                     value_list += values[1].Intensity.values.tolist()

                        y, bin_edges = np.histogram(v, bins=100)
                        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    #                     y_norm = y/np.max(y)
                        # Log scale
    #                     bin_log = np.log10(bin_centers + 1e-15)
    #                     ax.plot(bin_log, y_norm, label=values[0][31:])
    #                     ax.plot(bin_centers, y_norm, label=values[0][31:])
                        bin_log = np.log10(bin_centers + 1e-15)
                        ax.plot(bin_log, y, label=values[0][31:])
#                 ax.plot(bin_centers, y_norm, linestyle=ls, lw=lw, label=ll)
#             if method == 'a':
            ax.legend(fontsize=ft)
#             ax.set_xlim((0.15,0.4))
        #     ax.set_ylim((0,4500000))
        #     for side in ['top','bottom','left','right']:
        #         ax.spines[side].set_linewidth(4)
    #         [side.set_linewidth(4) for side in ax.spines.values()]
#             ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            filename = fig_dir + '/spot_int_histogram_method_' + method + '.pdf'
            ip.plt.show()
#             ip.plt.savefig(filename, transparent=True)
            ip.plt.close()

# %% codecell
# Build threshold curves
import numpy as np
threshs = np.linspace(0,1,2000)
curves = [[sp[sp.Intensity > t].shape[0] for t in threshs] for sp in sp_dist]
# curves_dict = imfn.get_nested_dict(keys, curves, [0,1,2])
# %% codecell
keys = [imfn.get_filename_keys(sn, filename_factors=factors) for sn in sample_names]
key_labels = [", ".join(k) for k in keys]
print(len(keys))
# %% codecell
# build dictionary
values = list(zip(sample_names, curves))
curves_dict = imfn.get_nested_dict(keys, values, [0,1,2])
# %% codecell
# Plot standard threshold curves
import image_plots as ip
I = ['a','b','c','d','e','f','g','h']
J = ['neg','pos']
K = ['1','2','3']

for moi in I:
    print('method',moi)
    fig, ax = ip.general_plot(col='w', dims=(10,5))
    for time in J:
        for fov in K:
            tup_list = curves_dict[moi][time][fov]
            for k, (sn, c) in tup_list:
                ax.plot(threshs, c, label=sn[31:], lw=3)
    ax.legend(fontsize=10)
#         ax.set_xlim(-0.05, 0.2)
    ip.plt.show()
    ip.plt.close()
# %% codecell
# Get slopes
curves_shift = [c[:1] + c[:-1] for c in curves]
slopes = [[y2-y1 for y1, y2 in zip(cs, c)] for cs, c in zip(curves_shift, curves)]
# %% codecell
# Find thresh with min slope and make new threshold values
slope_mins = [np.min(sl) for sl in slopes]
thresh_mins = [np.max(threshs[np.array(sl) == sm]) for sl, sm in zip(slopes, slope_mins)]
threshs_n = [threshs - tm for tm in thresh_mins]
# %% codecell
# build dictionary
values = list(zip(sample_names, curves, threshs_n))
norm_curves_dict = imfn.get_nested_dict(keys, values, [0,1,2])
# %% codecell
# Plot Normalized threshold curves
import image_plots as ip
I = ['a','b','c','d','e','f','g','h']
J = ['neg','pos']
K = ['1','2','3']

for moi in I:
    print('method',moi)
    fig, ax = ip.general_plot(col='w', dims=(10,5))
    for time in J:
        for fov in K:
            tup_list = norm_curves_dict[moi][time][fov]
            for k, (sn, c, t) in tup_list:
                ax.plot(t, c, label=sn[31:], lw=3)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 0.2)
    ip.plt.show()
    ip.plt.close()
# %% codecell
# Plot normalized threshold curves
import image_plots as ip

for moi in ['001','010','100']:
    print('moi',moi)
    for time in ['010','020']:
        print('time',time)
        fig, ax = ip.general_plot(col='w', dims=(10,5))
        tup_list_neg = curves_dict['000'][time]
        for k, (sn, c, t) in tup_list_neg:
            if not 'bg' in sn:
                ax.plot(t, c, label=sn[31:], lw=3)
        tup_list = curves_dict[moi][time]
        for k, (sn, c, t) in tup_list:
            ax.plot(t, c, label=sn[31:], lw=3)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.05, 0.2)
#         ax.set_ylim(0,500)
        ip.plt.show()
        ip.plt.close()
# %% codecell
