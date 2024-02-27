# Spatial Mapping of Mobile Genetic Elements and their Cognate Hosts in Complex Microbiomes 

This repository contains the specific code used to generate the figures in "Spatial Mapping of Mobile Genetic Elements and their Cognate Hosts in Complex Microbiomes"

## Acknowledgement

This code makes use of open source packages including `numpy`, `pandas`, `biopython`, `bioformats`, `javabridge`, `aicsimageio`, `scikit-image`, `scikit-learn`, `PySAL`, `OpenCV`, `scipy`, and `matplotlib`.

This code also makes use of [code](https://github.com/proudquartz/hiprfish) developed in [Shi, et al. 2020](https://doi.org/10.1038/s41586-020-2983-4). 

## Directories

`fig_1b_S1` - Optimization of single molecule MGE FISH 

`fig_1c` - Visualizing phage infection

`fig_2_S2_S3` - Mapping MGEs in oral plaque biofilms at high specificity

`fig_3_S4` - Combined taxonomic mapping and MGE mapping, and AMR gene distribution measurements

`fig_4_S4` - Identifying the host taxon of a novel plasmid

#### General use code

`conda_environments` - .yml files for building the required conda environments.

`functions` - .py files with commonly used custom functions. 

`rules` - .smk files used in Snakemake pipelines.'

`scripts` - .py files that can be executed in the command line.






