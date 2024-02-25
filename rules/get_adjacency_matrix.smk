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

rule get_adjacency_matrix:
    input:
        classif_filt_fn = config['output_dir'] + '/' + config['classif_filt_fmt'],
    output:
        adj_mat_fn =  config['output_dir'] + '/' + config['adj_mat_fmt'],
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/get_adjacency_matrix.py "
        "-c {params.config_fn} "
        "-pf {input.classif_filt_fn} "
        "-amf {output.adj_mat_fn} "

        # "-sp {input.seg_props_fn} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
        # seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
