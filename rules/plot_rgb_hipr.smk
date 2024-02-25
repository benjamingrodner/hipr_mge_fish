# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_01
# =============================================================================
"""
Plot several channels from the spectral image as an RGB image
"""
# =============================================================================
# Params

rule plot_rgb_hipr:
    input:
        reg_fn = config['output_dir'] + '/' + config['reg_fmt']
    output:
        rgb_fn = config['output_dir'] + '/' + config['rgb_fmt']
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
    shell:
        "python {params.pipeline_path}/scripts/HiPRFISH/plot_rgb_hipr.py "
        "-c {params.config_fn} "
        "-rf {input.reg_fn} "
        "-of {output.rgb_fn} "

        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
