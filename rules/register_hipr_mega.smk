# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2023_01_08
# =============================================================================
"""
Purpose: Used in pipeline for registering HiPRFISH images with MeGAFISH images
to correct x-y shifts. Upsamples HiPRFISH images so the resolution matches
MeGAFISH images, then uses phase cross correlation to overlay the two images.
"""
# =============================================================================
# Params
# seg_type = 'cell_seg'
# fn_mod = config_hipr[seg_type]['fn_mod']

hiprfmt = config['hipr']
megafmt = config['mega']

rule register_hipr_mega:
    input:
        hipr_max_fn = hipr_output_dir + '/' + config_hipr['max_fmt'],
        hipr_seg_fn = hipr_output_dir + '/' + config_hipr['seg_fmt'],
        hipr_seg_col_fn = hipr_output_dir + '/' + config_hipr['seg_filt_col_fmt'],
        mega_raw_fn = mega_output_dir + '/' + config_mega['raw_fmt'],
        mega_cell_seg_fns = mega_cell_seg_fmts,
        mega_spot_seg_fns = mega_spot_seg_fmts,
    output:
        hipr_max_resize_fn = config['output_dir'] + '/' + hiprfmt['max_resize'],
        hipr_seg_resize_fn = config['output_dir'] + '/' + hiprfmt['seg_resize'],
        hipr_props_resize_fn = config['output_dir'] + '/' + hiprfmt['props_resize'],
        hipr_seg_col_resize_fn = config['output_dir'] + '/' + hiprfmt['seg_col_resize'],
        mega_shift_vectors = config['output_dir'] + '/' + megafmt['shift_vectors'],
        mega_raw_shift_fn = config['output_dir'] + '/' + megafmt['raw_shift'],
        mega_cell_seg_shift_fns = mega_cell_seg_shift_fmts,
        mega_cell_props_shift_fns = mega_cell_props_shift_fmts,
        mega_spot_seg_shift_fns = mega_spot_seg_shift_fmts,
        mega_spot_props_shift_fns = mega_spot_props_shift_fmts,
    params:
        pipeline_path = config_hipr['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPR_MeGA/register_hipr_mega.py "
        "-c {params.config_fn} "
        "-hm {input.hipr_max_fn} "
        "-hs {input.hipr_seg_fn} "
        "-hsc {input.hipr_seg_col_fn} "
        "-mr {input.mega_raw_fn} "
        "-mcs {input.mega_cell_seg_fns} "
        "-mss {input.mega_spot_seg_fns} "
        "-hmr {output.hipr_max_resize_fn} "
        "-hsr {output.hipr_seg_resize_fn} "
        "-hpr {output.hipr_props_resize_fn} "
        "-hscr {output.hipr_seg_col_resize_fn} "
        "-msv {output.mega_shift_vectors} "
        "-mrs {output.mega_raw_shift_fn} "
        "-mcss {output.mega_cell_seg_shift_fns} "
        "-mcps {output.mega_cell_props_shift_fns} "
        "-msss {output.mega_spot_seg_shift_fns} "
        "-msps {output.mega_spot_props_shift_fns} "
