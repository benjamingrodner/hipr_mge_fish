# Figure 3b
Two volunteer control experiment.

## Data

Got images from 2021_08_25_plaqueamrintegrated imaging session.

Targeting genes: mefE, smeS, MexZ.

Sample hs has the genes in its metagenome.

Sample bmg doesn't have the genes in its metagenome.

## Segmentation

Followed nb_3b_segmentation.py jupyter notebook. Goes with the Snakemake_segment pipeline.

## Filtering

In nb_3b_filtering.py jupyter notebook, filtered objects by area, then split spots with multiple local maxima via Snakemake_multimax pipeline.

Then filtered spots by intensity. Then associated spots with cells via the Snakemake_spottocell pipeline.

## Plots

In nb_3b_plots.py jupyter notebook, plotted the spot count normalized by the number of cells or the area of cells.

Assessed the spatial autocorrelation of cells with spots using Moran's I statistic.

--------------------------------------------------------------------------------
## Update 7/27/22

### Manual background thresholding

Adopted nb_3b_backgroundmask.py to generate a spot mask with debris removed and most background spots removed.

Generated bar plots for each channel showing number of spot pixels after thresholding normalized by number of cell pixels. I ended up using the number of pixels without debris removed.

### Segmentation

nb_3b_segmentation_222707.py I ran only the spot segmentation

### Filtering

nb_3b_filtering_220722.py

I filtered cells for size, then I selected a fairly high SNR threshold for each spot channel, then I did spot to cell assignment. I ended up with ~0 spots in the negative control and ~200 spots in the positive control.

### plots

nb 3b_plots.py

Generated Moran's I and black white join counts statistics. 
