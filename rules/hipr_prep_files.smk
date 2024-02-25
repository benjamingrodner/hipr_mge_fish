# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2023_01_05
# =============================================================================
"""
Read czi spectral images and prepare npy files for segmentation and
classification
"""
# =============================================================================

rule hipr_prep_files:
    input:
        get_raw_filenames
    output:
        reg_fn = config['output_dir'] + '/' + config['reg_fmt'],
        max_fn = config['output_dir'] + '/' + config['max_fmt'],
        sum_fn = config['output_dir'] + '/' + config['sum_fmt']
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
        script_path = config['prep']['script']
    shell:
        "python {params.pipeline_path}/{params.script_path} "
        "-f {input} "
        "-c {params.config_fn} "
        "-r {output.reg_fn} "
        "-m {output.max_fn} "
        "-s {output.sum_fn} "
    # conda:
    #     config['conda_env_fn']
