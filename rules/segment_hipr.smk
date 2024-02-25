# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_01
# =============================================================================
"""
Segment the cells in a 16s rRNA stained image
"""
# =============================================================================
# Params
seg_type = 'cell_seg'
fn_mod = config[seg_type]['fn_mod']

rule segment_hipr:
    input:
        sum_fn = config['output_dir'] + '/' + config['sum_fmt']
        # config['output_dir'] + '/' + config[config['segmentation']['seg_in_file']]
    output:
        seg_fn = (config['output_dir'] + '/' + config['seg_fmt']),
        process_fn = (config['output_dir'] + '/' + config['seg_process_fmt'])
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
        seg_type = seg_type
    resources: mem_gb=config['seg_mem_gb']
    shell:
        "python {params.pipeline_path}/scripts/segment.py "
        "{input} {output.seg_fn} "
        "-cfn {params.config_fn} "
        "-st {params.seg_type} "
        "-pfn {output.process_fn} "
        "-sn {wildcards.sample_name} "
        "-ch all "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
