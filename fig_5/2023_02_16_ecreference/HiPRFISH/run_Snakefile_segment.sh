snakemake -s Snakefile_segment --configfile config_hipr.yaml -j 23 --resources mem_gb=100 -pn -R classify_spectra