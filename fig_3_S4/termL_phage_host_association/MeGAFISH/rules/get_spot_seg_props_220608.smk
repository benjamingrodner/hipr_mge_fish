# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_01
# =============================================================================
"""
Given segmented spots image, find local maxima and split spots with multiple
local maxima
"""
# =============================================================================
# Params
seg_type = 'spot_seg'
fn_mod = config[seg_type]['fn_mod']

rule get_spot_seg_props_220608:
    input:
        raw = config['output_dir'] + '/raw_npy/{sample_name}.npy',
        seg = (config['output_dir'] + '/' + seg_type + '/{sample_name}/'
               + '{sample_name}_chan_{channel_spot}' + fn_mod + '.npy'),
    output:
        seg_props = (config['output_dir'] + '/'
                     + seg_type + '_props' + '/{sample_name}/'
                     + '{sample_name}_chan_{channel_spot}' + fn_mod + '_props.csv'),
        # locmax = (config['output_dir'] + '/'
        #           + seg_type + '_props' + '/{sample_name}/'
        #           + '{sample_name}_chan_{channel_spot}' + fn_mod + '_locmax.npy'),
        # locmax_props = (config['output_dir'] + '/'
        #                 + seg_type + '_props' + '/{sample_name}/'
        #                 + '{sample_name}_chan_{channel_spot}'
        #                 + fn_mod + '_locmax_props.csv'),
        # multimax_table = (config['output_dir'] + '/'
        #                   + seg_type + '_props' + '/{sample_name}/'
        #                   + '{sample_name}_chan_{channel_spot}'
        #                   + fn_mod + '_multimax.csv'),
        # seg_split_props = (config['output_dir'] + '/'
        #                    + seg_type + '_props' + '/{sample_name}/'
        #                    + '{sample_name}_chan_{channel_spot}'
        #                    + fn_mod + '_split_props.csv'),
    params:
        config = config_fn,
        pipeline_path = config['pipeline_path'],
        seg_type = seg_type
    shell:
        "python {params.pipeline_path}/scripts/get_seg_props_220608.py "
        "-cfn {params.config} "
        "-st {params.seg_type} "
        "-r {input.raw} "
        "-s {input.seg} "
        "-ch {wildcards.channel_spot} "
        "-spfn {output.seg_props} "
        # "-lmfn {output.locmax} "
        # "-lmpfn {output.locmax_props} "
        # "-mtfn {output.multimax_table} "
        # "-sspfn {output.seg_split_props} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
