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

rule get_cell_seg_spot_props_220608:
    input:
        raw = config['output_dir'] + '/raw_npy/{sample_name}.npy',
        seg = (config['output_dir'] + '/' + seg_type + '/{sample_name}/'
               + '{sample_name}_chan_{channel_cell}' + fn_mod + '.npy'),
    output:
        seg_props = (config['output_dir'] + '/'
                        + seg_type + '_props/{sample_name}/'
                        + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                        + fn_mod + '_props.csv'),
        locmax = (config['output_dir'] + '/'
                  + seg_type + '_props' + '/{sample_name}/'
                  + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                  + fn_mod + '_locmax.npy'),
        locmax_props = (config['output_dir'] + '/'
                        + seg_type + '_props' + '/{sample_name}/'
                        + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                        + fn_mod + '_locmax_props.csv'),
        multimax_table = (config['output_dir'] + '/'
                          + seg_type + '_props' + '/{sample_name}/'
                          + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                          + fn_mod + '_multimax.csv'),
        seg_split_props = (config['output_dir'] + '/'
                           + seg_type + '_props' + '/{sample_name}/'
                           + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                           + fn_mod + '_split_props.csv'),
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
        "-m True "
        "-lmfn {output.locmax} "
        "-lmpfn {output.locmax_props} "
        "-mtfn {output.multimax_table} "
        "-sspfn {output.seg_split_props} "

        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
