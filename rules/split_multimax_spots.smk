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

rule split_multimax_spots:
    input:
        raw = config['output_dir'] + '/' + config['raw_fmt'],
        seg = (config['output_dir'] + '/' + config['spot_seg_area_filt_fmt']),
        seg_props = (config['output_dir'] + '/'
                     + config['spot_props_area_filt_fmt']),
    output:
        locmax = (config['output_dir'] + '/'
                  + config['spot_props_locmax_fmt']),
        locmax_props = (config['output_dir'] + '/'
                        + config['spot_props_locmax_props_fmt']),
        multimax_table = (config['output_dir'] + '/'
                          + config['spot_props_multimax_fmt']),
        seg_split = (config['output_dir'] + '/'
                            + config['spot_seg_max_split_fmt']),
        seg_split_props = (config['output_dir'] + '/'
                            + config['spot_props_max_split_fmt'])
    params:
        config = config_fn,
        pipeline_path = config['pipeline_path'],
        seg_type = seg_type
    shell:
        "python {params.pipeline_path}/scripts/split_multimax_spots.py "
        "-cfn {params.config} "
        "-st {params.seg_type} "
        "-r {input.raw} "
        "-s {input.seg} "
        "-ch {wildcards.spot_chan} "
        "-spfn {input.seg_props} "
        "-lmfn {output.locmax} "
        "-lmpfn {output.locmax_props} "
        "-mtfn {output.multimax_table} "
        "-ssfn {output.seg_split} "
        "-sspfn {output.seg_split_props} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
