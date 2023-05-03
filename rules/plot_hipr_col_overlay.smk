# Snakemake rule
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2023_01_08
# =============================================================================
"""
Purpose: Used in pipeline for registering HiPRFISH images with MeGAFISH images
to plot spots on colored HiPRFISH images.
"""
# =============================================================================
# Params
# seg_type = 'cell_seg'
# fn_mod = config_hipr[seg_type]['fn_mod']

hiprfmt = config['hipr']
megafmt = config['mega']

rule plot_hipr_col_overlay:
    input:
        mega_spot_props_shift_fns = mega_spot_props_shift_fmts,
        hipr_seg_col_resize_fn = config['output_dir'] + '/' + hiprfmt['seg_col_resize'],
    output:
        overlay_hipr_fns = overlay_hipr_col_fmts,
    params:
        pipeline_path = config_hipr['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPR_MeGA/plot_hipr_col_overlay.py "
        "-c {params.config_fn} "
        "-msps {input.mega_spot_props_shift_fns} "
        "-hscr {input.hipr_seg_col_resize_fn} "
        "-oh {output.overlay_hipr_fns} "