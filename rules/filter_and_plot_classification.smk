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

rule filter_and_plot_classification:
    input:
        seg_fn = config['output_dir'] + '/' + config['seg_fmt'],
        reg_fn = config['output_dir'] + '/' + config['reg_fmt'],
        classif_fn = config['output_dir'] + '/' + config['props_classif_fmt'],
    output:
        plot_spec_raw_fn = config['output_dir'] + '/' + config['plot_spec_raw_fmt'],
        plot_spec_norm_fn = config['output_dir'] + '/' + config['plot_spec_norm_fmt'],
        plot_nndist_fn = config['output_dir'] + '/' + config['plot_nndist_fmt'],
        seg_col_fn = config['output_dir'] + '/' + config['seg_col_fmt'],
        plot_seg_col_fn = config['output_dir'] + '/' + config['plot_seg_col_fmt'],
        seg_filt_col_fn = config['output_dir'] + '/' + config['seg_filt_col_fmt'],
        plot_seg_filt_col_fn = config['output_dir'] + '/' + config['plot_seg_filt_col_fmt'],
        filt_summ_fn = config['output_dir'] + '/' + config['filt_summ_fmt'],
        classif_filt_fn = config['output_dir'] + '/' + config['classif_filt_fmt'],
        plot_classif_legend_fn = config['output_dir'] + '/' + config['plot_classif_legend_fmt'],
        plot_classif_filt_legend_fn = config['output_dir'] + '/' + config['plot_classif_filt_legend_fmt'],
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/filter_and_plot_classification.py "
        "-c {params.config_fn} "
        "-sf {input.seg_fn} "
        "-rf {input.reg_fn} "
        "-cf {input.classif_fn} "
        "-psrf {output.plot_spec_raw_fn} "
        "-psnf {output.plot_spec_norm_fn} "
        "-pndf {output.plot_nndist_fn} "
        "-scf {output.seg_col_fn} "
        "-pscf {output.plot_seg_col_fn} "
        "-sfcf {output.seg_filt_col_fn} "
        "-psfcf {output.plot_seg_filt_col_fn} "
        "-fsf {output.filt_summ_fn} "
        "-cff {output.classif_filt_fn} "
        "-pclf {output.plot_classif_legend_fn} "
        "-pcflf {output.plot_classif_filt_legend_fn} "

        # "-sp {input.seg_props_fn} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
        # seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
