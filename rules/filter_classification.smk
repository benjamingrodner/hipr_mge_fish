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

rule filter_classification:
    input:
        classif_fn = config['output_dir'] + '/' + config['props_classif_fmt'],
    output:
        plot_spec_raw_fn = config['output_dir'] + '/' + config['plot_spec_raw_fmt'],
        plot_cell_maxint_fn = config['output_dir'] + '/' + config['plot_cell_maxint_fmt'],
        plot_nndist_fn = config['output_dir'] + '/' + config['plot_nndist_fmt'],
        filt_summ_fn = config['output_dir'] + '/' + config['filt_summ_fmt'],
        classif_filt_fn = config['output_dir'] + '/' + config['classif_filt_fmt'],
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/filter_classification.py "
        "-c {params.config_fn} "
        "-cf {input.classif_fn} "
        "-psrf {output.plot_spec_raw_fn} "
        "-pcmf {output.plot_cell_maxint_fn} "
        "-pndf {output.plot_nndist_fn} "
        "-fsf {output.filt_summ_fn} "
        "-cff {output.classif_filt_fn} "

        # "-sp {input.seg_props_fn} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
        # seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
