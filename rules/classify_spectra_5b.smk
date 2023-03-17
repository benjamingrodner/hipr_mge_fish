# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_12_13
# =============================================================================
"""
"""
# =============================================================================
# Params
# seg_type = 'cell_seg'
# fn_mod = config[seg_type]['fn_mod']

rule classify_spectra:
    input:
        seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt'],
        reg_fn = config['output_dir'] + '/' + config['reg_fmt'],
        seg_fn = config['output_dir'] + '/' + config['seg_fmt'],
        svc_fn = svc_fn
    output:
        classif_fn = config['output_dir'] + '/' + config['props_classif_fmt']
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/classify_spectra_5b.py "
        "-c {params.config_fn} "
        "-sp {input.seg_props_fn} "
        "-sf {input.seg_fn} "
        "-rf {input.reg_fn} "
        "-cf {output} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
