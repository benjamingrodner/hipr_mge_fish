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

rule plot_adjacency_graph:
    input:
        adj_mat_fn =  config['output_dir'] + '/' + config['adj_mat_fmt'],
    output:
        plot_adj_graph_fn =  config['output_dir'] + '/' + config['plot_adj_graph_fmt'],
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/plot_adjacency_graph.py "
        "-c {params.config_fn} "
        "-amf {input.adj_mat_fn} "
        "-pag {output.plot_adj_graph_fn} "

        # "-sp {input.seg_props_fn} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
        # seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
