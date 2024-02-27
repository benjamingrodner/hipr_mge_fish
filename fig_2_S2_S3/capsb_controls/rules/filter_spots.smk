# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given spot segmentation and local maxima properties tables and filter values,
remove dim spots, and oversize spots
"""
# =============================================================================
# Params
seg_type = 'spot_seg'
fn_mod = config[seg_type]['fn_mod']

rule filter_spots:
    input:
        # cell = (config['output_dir'] + '/cell_seg/{sample_name}_cell_seg.npy'),
        spot_seg = (config['output_dir'] + '/spot_seg/{sample_name}/'
               + '{sample_name}_chan_{channel_spot}_cell_seg.npy'),
        spot_props = (config['output_dir'] + '/' + seg_type + '_props' + '/{sample_name}/'
                        + '{sample_name}_chan_{channel_spot}' + fn_mod + '_props.csv'),
        # max = (config['output_dir'] + '/spot_seg_props/{sample_name}_max_props.csv')
        max = (config['output_dir'] + '/spot_seg_props/{sample_name}/'
                        + '{sample_name}_chan_{channel_spot}' + fn_mod + '_max_props.csv'),
        filt = (config['output_dir'] + '/spot_seg_props/{sample_name}/'
                        + '{sample_name}_chan_{channel_spot}' + fn_mod + '_max_props.csv')
    output:
         # (config['output_dir'] + '/spot_analysis/{sample_name}_max_props_cid.csv')
        (config['output_dir'] + '/spot_filtered/{sample_name}/'
                + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                + fn_mod + '_max_props_filt.csv')
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
    shell:
        "python {params.pipeline_path}/scripts/filter_spots.py "
        "-cfn {params.config_fn} "
        "-ch {wildcards.channel_spot} "
        "-ss {input.spot_seg} "
        "-ssp {input.spot_props} "
        "-mp {input.max} "
        "-fv {input.filt} "
        "-mpf {output} "

        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
