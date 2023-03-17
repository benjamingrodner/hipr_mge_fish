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

rule plot_classification_unfilt:
    input:
        seg_fn = config['output_dir'] + '/' + config['seg_fmt'],
        classif_fn = config['output_dir'] + '/' + config['props_classif_fmt'],
    output:
        seg_col_fn = config['output_dir'] + '/' + config['seg_col_fmt'],
        plot_seg_col_fn = config['output_dir'] + '/' + config['plot_seg_col_fmt'],
        plot_classif_legend_fn = config['output_dir'] + '/' + config['plot_classif_legend_fmt'],
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/plot_classification.py "
        "-c {params.config_fn} "
        "-sf {input.seg_fn} "
        "-cf {input.classif_fn} "
        "-scf {output.seg_col_fn} "
        "-pscf {output.plot_seg_col_fn} "
        "-pclf {output.plot_classif_legend_fn} "

        # "-sp {input.seg_props_fn} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
        # seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
