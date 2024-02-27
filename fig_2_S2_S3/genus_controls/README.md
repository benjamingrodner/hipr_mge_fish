# Figure 3e

Genus DNA probes

Goal is to count the number of cells with the genus specific 16s rRNA stain that also have the genus specific 16s DNA stain.

## Data

2022_03_27_plaquegenuscontrol_probes_laut_fov_2_Airyscan_Processing.czi

2022_03_27_plaquegenuscontrol_probes_strep_fov_1_Airyscan_Processing.czi

## Background removal

Select background thresholds and remove debris. Chose different thresholds for the two types.

## Segmentation

Added more gaussian blurring and denoising to the cell seg than normal. Only paid attention to the universal stain for cell segmentation.

## filtering

Area filtering not too important, not a lot of debris for the cells. Did not filter any SNR for the spots, there is just too much off target binding in this dataset. 
