# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_01
# =============================================================================
"""
Read an image file using bioformats and write to numpy array
"""
# =============================================================================

rule write_files_to_npy:
    input:
        config['input_dir'] + '/{sample_name}' + config['input_ext']
    output:
        config['output_dir'] + '/raw_npy/{sample_name}.npy'
    params:
        pipeline_path = config['pipeline_path']
    shell:
        "python {params.pipeline_path}/scripts/write_to_npy.py "
        "{input} {output}"
    # conda:
    #     config['conda_env_fn']
