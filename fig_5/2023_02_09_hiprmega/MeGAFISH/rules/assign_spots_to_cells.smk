# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_02_03
# =============================================================================
"""
Given cell segmentation and spot segmenation properties tables, assign spots to
cells
"""
# =============================================================================
# Params
seg_type = 'cell_seg'
fn_mod = config[seg_type]['fn_mod']

rule assign_spots_to_cells:
    input:
        # cell = (config['output_dir'] + '/cell_seg/{sample_name}_cell_seg.npy'),
        cell = (config['output_dir'] + '/cell_seg/{sample_name}/'
               + '{sample_name}_chan_{channel_cell}_cell_seg.npy'),
        seg_props_fn = (config['output_dir'] + '/' + seg_type + '_props/{sample_name}/'
                        + '{sample_name}_chan_{channel_cell}' + fn_mod + '_props.csv'),
        # max = (config['output_dir'] + '/spot_seg_props/{sample_name}_max_props.csv')
        raw_fn = config['output_dir'] + '/raw_npy/{sample_name}.npy',
        max = (config['output_dir'] + '/spot_filtered/{sample_name}/'
                + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                + fn_mod + '_max_props_filt.csv')
    output:
         # (config['output_dir'] + '/spot_analysis/{sample_name}_max_props_cid.csv')
        max_props_cid = (config['output_dir'] + '/spot_analysis/{sample_name}/'
                + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                + '_max_props_cid.csv')
        cell_props_spot_max_fn = (config['output_dir'] + '/' + seg_type
                + '_props_spot_max/{sample_name}/'
                + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                + fn_mod + '_props_spot_max.csv'),
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
        raw_type = 'spot_seg'
    shell:
        "python {params.pipeline_path}/scripts/assign_spots_to_cells.py "
        "-cfn {params.config_fn} "
        "-cs {input.cell} "
        "-cp {input.seg_props_fn} "
        "-cpm {outputs.cell_props_spot_max_fn} "
        "-rf {input.raw_fn} "
        "-ch {wildcards.channel_spot} "
        "-mp {input.max} "
        "-mpc {output.max_props_cid} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
