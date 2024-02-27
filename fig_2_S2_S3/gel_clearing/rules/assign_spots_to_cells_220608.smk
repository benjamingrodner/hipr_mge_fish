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
fn_mod_s = config['spot_seg']['fn_mod']
fn_mod_c = config['cell_seg']['fn_mod']


rule assign_spots_to_cells_220608:
    input:
        # cell = (config['output_dir'] + '/cell_seg/{sample_name}_cell_seg.npy'),
        raw = config['output_dir'] + '/raw_npy/{sample_name}.npy',
        spot_seg = (config['output_dir'] + '/spot_seg/{sample_name}/'
                    + '{sample_name}_chan_{channel_spot}' + fn_mod_s + '.npy'),
        cell_seg = (config['output_dir'] + '/cell_seg/{sample_name}/'
                    + '{sample_name}_chan_{channel_cell}' + fn_mod_c + '.npy'),
        spot_props = (config['output_dir'] + '/'
                           + 'spot_seg_props' + '/{sample_name}/'
                           + '{sample_name}_chan_{channel_spot}'
                           + fn_mod_s + '_split_props.csv'),
    output:
         # (config['output_dir'] + '/spot_analysis/{sample_name}_max_props_cid.csv')
        spot_cell_id = (config['output_dir'] + '/spot_analysis/{sample_name}/'
                + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                + '_spot_cid.csv'),
        spot_cell_id_demult = (config['output_dir'] + '/spot_analysis/{sample_name}/'
                + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                + '_spot_cid_demult.csv'),
        spot_props_cid = (config['output_dir'] + '/spot_analysis/{sample_name}/'
                + '{sample_name}_cellchan_{channel_cell}_spotchan_{channel_spot}'
                + fn_mod_s + '_props_cid.csv'),
    params:
        config_fn = config_fn,
        pipeline_path = config['pipeline_path'],
    shell:
        "python {params.pipeline_path}/scripts/assign_spots_to_cells_220608.py "
        "-cfn {params.config_fn} "
        "-st spot_seg "
        "-r {input.raw} "
        "-ss {input.spot_seg} "
        "-cs {input.cell_seg} "
        "-sp {input.spot_props} "
        "-ch {wildcards.channel_spot} "
        "-scid {output.spot_cell_id} "
        "-scm {output.spot_cell_id_demult} "
        "-sspc {output.spot_props_cid} "
        # "-pp {params.pipeline_path} "
    # conda:
    #     config['conda_env_fn']
