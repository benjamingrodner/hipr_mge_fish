# %% md

# # Figure 3b: Image analysis

# Ran after running nb_3b_segmentation.py. Used the "pysal" conda environment.


# %% md
# ==============================================================================
# ## Load images and segmentation data
# ==============================================================================

# %% md

# Load raw images

# %% md

# Load segmentations

# %% md

# %% md

# Load segmentation properties tables

# %% md

# Show raw images and segmentations

# %% md
# ==============================================================================
# ## Clean up the MGE FISH segmentations
# ==============================================================================

# ### Remove off-target binding to debris in segmentations by area thresholding

# Plot object areas as a histogram

# %% md

# Plot number of objects as a function of area threshold value and select threshold values for each channel.

# %% md

# Add boolean area thresholding column to segmentation properties tables and create an area-filtered table

# %% md

# ### Remove background spots by intensity thresholding

# plot the area filtered table for object max intensity as a histogram

# %% md

# Plot number of objects as a function of intensity threshold value and select a threshold for each channel.

# %% md

# Add boolean intensity thresholding column to segmentation properties tables and create an intensity/area-filtered table

# %% md
# ==============================================================================
# ## Calculate analysis values and plot
# ==============================================================================

# ### Plot spots per cell

# Plot average spots per cell with the selected threshold

# %% md

# Plot average spots per cell as a function of intensity threshold

# %% md

# ### Global Moran's I statistics

# Get spatial weights matrix

# %% codecell
for sn in sample_names:
    cis_fn = cell_info_spot_count_fnt.format(sn)
    seg_info = pd.read_csv(cis_fn)
    points = seg_info.loc[:,['centroid-0','centroid-1']].values
    w = Voronoi(points)
    w_fn = weights_fnt.format(sn)
    sv = saf.save_weights(w, w_fn)

# %% md

# Calculate global moran's I statistics and save values

# %% codecell
for sn in sample_names:
    w_fn = weights_fnt.format(sn)
    w = saf.read_weights(w_fn)
    w.transform = 'r'
    for scn in spot_channels_names:
        cis_fn = cell_info_spot_count_fnt.format(sn)
        cis = pd.read_csv(cis_fn)
        y = cis[scn + moran_i_val]
        # Moran's I calculation, save the I stat and the p value for the simulation-based null
        mi = esda.moran.Moran(y,w)
        I.append(mi.I)
        p_sim.append(mi.p_sim)


# %% md

# Plot Moran's I simulation and statistic

# %% codecell
lw=1
ft = 5
dims = (0.9843,0.5905512)
col = 'k'
colors = apl.get_cmap_listed('tab10')
line_colors = [colors[0],colors[1]]
xlabel = ''
xticks = (1,1.7)
xlims = (0.65,2.05)
ylabel = ''
yticks = []
pad = 0.2

for sn, mi, l_col in zip(sample_names, morans_is, line_colors):
    for c in spot_channels:
        fig, ax = afn.general_plot(col=col, dims=dims, lw=lw, ft=ft, pad=pad)
        ax.set_xticks(xticks)
        ax.set_xticklabels([],visible=False)
        ax.set_xlim(xlims[0],xlims[1])
        ax.set_yticks(yticks)

        apl.plot_morans_i_sim(ax, mi, col=l_col)
        output_fn = fig_dir + '/' + sn + '_moransI_chan_' + str(c)
        apl.save_png_pdf(output_fn, bbox_inches=False)
