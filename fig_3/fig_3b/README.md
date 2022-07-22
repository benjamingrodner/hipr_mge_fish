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
