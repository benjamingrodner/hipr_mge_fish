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
hmfmt = config['hipr_mega']

rule spot_taxa_association:
    input:
        mega_spot_props_shift_fns = mega_spot_props_shift_fmts,
        hipr_props_resize_fn = config['output_dir'] + '/' + hiprfmt['props_resize'],
        hipr_props_classif_fn = hipr_output_dir + '/' + config_hipr['classif_filt_fmt'],
    output:
        dict_tax_count_sim_fns = dict_tax_count_sim_fmts,
        tax_prob_fns = tax_assoc_prob_fmts,
    params:
        pipeline_path = config_hipr['pipeline_path'],
        config_fn = config_fn
    shell:
        "python {params.pipeline_path}/scripts/HiPR_MeGA/spot_taxa_association.py "
        "-c {params.config_fn} "
        "-spf {input.mega_spot_props_shift_fns} "
        "-cpf {input.hipr_props_resize_fn} "
        "-cbf {input.hipr_props_classif_fn} "
        "-dtcs {output.dict_tax_count_sim_fns} "
        "-tpf {output.tax_prob_fns} "
