# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given spot segmentation get the object properties table, but using each spot
channel as an intensity image. Also find local maxima in each object.
"""
# =============================================================================
# Params
seg_type = 'cell_seg'
fn_mod = config[seg_type]['fn_mod']

rule get_cell_seg_spot_props:
    input:
        raw_fn = config['output_dir'] + '/raw_npy/{sample_name}.npy',
        seg_fn = (config['output_dir'] + '/' + seg_type + '/{sample_name}/'
               + '{sample_name}_chan_{channel_cell}' + fn_mod + '.npy'),
    output:
        seg_props_fn = (config['output_dir'] + '/'
                        + seg_type + '_props/{sample_name}/'
                        + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                        + fn_mod + '_props.csv'),
        max_props_fn = (config['output_dir'] + '/'
                        + seg_type + '_props/{sample_name}/'
                        + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                        + fn_mod + '_max_props.csv'),
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
        seg_type = seg_type
    shell:
        "python {params.pipeline_path}/scripts/get_seg_props.py "
        "-cfn {params.config_fn} "
        "-r {input.raw_fn} "
        "-s {input.seg_fn} "
        "-st {params.seg_type} "
        "-sp {output.seg_props_fn} "
        "-m True "
        "-mp {output.max_props_fn} "
        "-ch {wildcards.channel_spot} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
