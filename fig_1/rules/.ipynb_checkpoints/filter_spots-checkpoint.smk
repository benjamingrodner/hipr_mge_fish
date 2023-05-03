rule filter_spots:
    input:
        spot_props_cell_id = out_dir + '/{sample}' + seg_fname_mod_spot + spot_cell_id_ext
    output:
        spot_props_filtered = out_dir + '/{sample}' + seg_fname_mod_spot + props_filter_ext,
        filter_stats = out_dir + '/{sample}' + seg_fname_mod_spot + filter_stats_ext,
#         filter_steps_figure = [out_dir + '/{sample}' + seg_fname_mod_spot + ext 
#                                for ext in filter_steps_figure_exts]
    params:
        functions_path = functions_path,
        sample = "{sample}",
        seg_fname_mod = seg_fname_mod_spot,
        in_ext = spot_cell_id_ext,
        out_ext = props_filter_ext,
        stats_ext = filter_stats_ext,
        image_processing_dir = out_dir,
        intensity_threshold = intensity_threshold,
        dist_thresh = dist_thresh,
        area_thresh = area_thresh,
        raw_ext = raw_ext,
        seg_ext = seg_ext,
        flat_field_filename = flat_field_filename,
        ff_index = flat_field_index,
        fig_exts = filter_steps_figure_exts,
        dims = dims_filter_figure,
        lims = clims_spot,
        spot_channel_index = spot_channel_index

    shell:
        "python filter_spots.py  "
        "-fnp {params.functions_path} "
        "-snm {params.sample} "
        "-sfm {params.seg_fname_mod} "        
        "-iext {params.in_ext} "        
        "-oext {params.out_ext} "        
        "-stext {params.stats_ext} "        
        "-ipd {params.image_processing_dir} "        
        "-ith {params.intensity_threshold} "        
        "-dth {params.dist_thresh} "        
        "-ath {params.area_thresh} "        
        "-rext {params.raw_ext} "        
        "-sext {params.seg_ext} "        
        "-ffn {params.flat_field_filename} "        
        "-ffi {params.ff_index} "        
        "-fexts {params.fig_exts} "        
        "-dims {params.dims} "        
        "-lims {params.lims} "        
        "-sci {params.spot_channel_index}"        
