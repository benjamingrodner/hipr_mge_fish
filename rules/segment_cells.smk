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

rule segment_cells:
    input:
        config['output_dir'] + '/raw_npy/{sample_name}.npy'
    output:
        seg_fn = (config['output_dir'] + '/' + seg_type + '/{sample_name}/'
               + '{sample_name}_chan_{channel_cell}' + fn_mod + '.npy'),
        process_fn = (config['output_dir'] + '/' + seg_type + '/{sample_name}/'
                   + '{sample_name}_chan_{channel_cell}' + fn_mod
                   + config['seg_process_ext'])
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
        seg_type = seg_type
    shell:
        "python {params.pipeline_path}/scripts/segment.py "
        "{input} {output.seg_fn} "
        "-cfn {params.config_fn} "
        "-st {params.seg_type} "
        "-pfn {output.process_fn} "
        "-ch {wildcards.channel_cell} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
