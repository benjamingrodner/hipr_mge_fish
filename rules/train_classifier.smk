# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2023_02_13
# =============================================================================
"""
"""
# =============================================================================
# Params
# seg_type = 'cell_seg'
# fn_mod = config[seg_type]['fn_mod']

rule train_classifier:
    input:
        probe_design_filename
    output:
        svc_fn
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/hiprfish_imaging_train_reference_5b.py "
        "-c {params.config_fn} "

        # "-sp {input.seg_props_fn} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
        # seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
