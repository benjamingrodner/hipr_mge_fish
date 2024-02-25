# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given cell segmentation get the object properties table
"""
# =============================================================================
# Params
# seg_type = 'cell_seg'
# fn_mod = config[seg_type]['fn_mod']

rule get_cell_spectra:
    input:
        raw_fn = config['output_dir'] + '/' + config['reg_fmt'],
        seg_fn = config['output_dir'] + '/' + config['seg_fmt']
    output:
        seg_props_fn = config['output_dir'] + '/' + config['seg_props_fmt']
    params:
        pipeline_path = config['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/get_cell_spectra.py "
        "-cfn {params.config_fn} "
        "-r {input.raw_fn} "
        "-s {input.seg_fn} "
        "-sp {output} "
        # "-ch {wildcards.channel_cell} "
        # "-st {params.seg_type} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']

    # seg_type = seg_type
