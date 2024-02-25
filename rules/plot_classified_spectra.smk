# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2023_01_08
# =============================================================================
"""
"""
# =============================================================================
# Params
# seg_type = 'cell_seg'
# fn_mod = config[seg_type]['fn_mod']

rule plot_classified_spectra:
    input:
        classif_filt_fn = config['output_dir'] + '/' + config['classif_filt_fmt'],
    output:
        plot_spec_classif_dir = directory(config['output_dir']
                + '/' + config['plot_spec_classif_dir']),
        plot_spec_classif_complete_fn = (config['output_dir']
                + '/' + config['plot_spec_classif_dir']
                + '/' + config['plot_spec_classif_complete_fmt']),
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/plot_classified_spectra.py "
        "-c {params.config_fn} "
        "-pf {input.classif_filt_fn} "
        "-sn {wildcards.sample_name} "
        "-od {output.plot_spec_classif_dir} "
        "-cf {output.plot_spec_classif_complete_fn} "

        # "-sp {input.seg_props_fn} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
        # seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
