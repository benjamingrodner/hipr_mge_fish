import pandas as pd
import numpy as np
import sys
import argparse




def main():
    # Load args
    parser = argparse.ArgumentParser('')
    parser.add_argument('-fnp', '--functions_path', dest ='functions_path', 
                        default='./', type = str, help = '')    
    
    parser.add_argument('-snm', '--sample_name', dest ='sample_name', 
                        type = str, help = '')
    parser.add_argument('-sfm', '--seg_fname_mod', dest ='seg_fname_mod', 
                        type = str, help = '')
    parser.add_argument('-iext', '--in_ext', dest ='in_ext', default='_cell_id.csv', 
                        type = str, help = '')
    parser.add_argument('-oext', '--out_ext', dest ='out_ext', default='_props_filtered.csv',
                        type = str, help = '')
    parser.add_argument('-stext', '--stats_ext', dest ='stats_ext', default='_filter_stats.csv',
                        type = str, help = '')
    parser.add_argument('-ipd', '--image_processing_dir', dest ='image_processing_dir', 
                        default='../image_processing', type = str, help = '')
    parser.add_argument('-ith', '--intensity_threshold', dest='intensity_threshold', 
                        default=0.05, type=float, help='')
    parser.add_argument('-dth', '--dist_thresh', dest='dist_thresh', 
                        default=10, type=float, help='')
    parser.add_argument('-ath', '--area_thresh', dest='area_thresh', 
                        default=200, type=float, help='')

    parser.add_argument('-rext', '--raw_ext', dest='raw_ext', 
                        default='.npy', type=str, help='')
    parser.add_argument('-sext', '--seg_ext', dest='seg_ext', 
                        default='.npy', type=str, help='')
    parser.add_argument('-ffn', '--flat_field_filename', dest='flat_field_filename', 
                        type=str, help='')
    parser.add_argument('-ffi', '--ff_index', dest='ff_index', 
                        default=0, type=int, help='')
    parser.add_argument('-fexts', '--fig_exts', dest='fig_exts', 
                        default=['_filter_steps.png','_filter_steps.pdf'], 
                        nargs='+', type=str, help='')
    parser.add_argument('-dims', '--dims', dest='dims', 
                        default=[10,2], nargs='+', type=float, help='')
    parser.add_argument('-lims', '--lims', dest='lims', 
                        default=['min','max'], nargs='+', type=str, help='')
    parser.add_argument('-sci', '--spot_channel_index', dest='spot_channel_index', 
                        default=1, type=int, help='')
    args = parser.parse_args()
    
    # Load functions
    sys.path.append(args.functions_path)
    import image_plots as ip
    import image_functions as ifn
    def filter_raw(raw, seg, props, id_column='ID'):
        retained_ids = props[id_column].values
        mask = ifn.spot_filter_mask(seg=seg, retained_ids=retained_ids)
        return raw * mask
    
    ###
    # Filter the spot properties table
    ###
    sn = args.sample_name
    seg_fname_mod = args.seg_fname_mod
    in_ext = args.in_ext
    out_ext = args.out_ext
    stats_ext = args.stats_ext
    out_dir = args.image_processing_dir
    int_thresh_normalized = args.intensity_threshold
    dist_thresh = args.dist_thresh
    area_thresh = args.area_thresh
#     print('0\n')
    # Get spot props with cell assignment
    spot_props_filename = '{}/{}{}{}'.format(out_dir, sn, seg_fname_mod, in_ext)
    spot_props = pd.read_csv(spot_props_filename, index_col=0)
    # Filter based on distance
    filter_distance = spot_props.loc[(spot_props.dist < dist_thresh),:]
    
    ### Threshold curve normalized filter
    # Take the distance filtered spots and count spots at different thresholds
    threshs = np.linspace(0,1,100)
    t_curve = [filter_distance.loc[(filter_distance.Intensity > t),:].shape[0] 
                for t in threshs]
    # Plot threshold curve
    fig, ax = ip.general_plot(col='w')
    ax.plot(threshs, t_curve)
    curve_ext = '_int_thresh_curve.png'
    t_curve_filename = '{}/{}{}{}'.format(out_dir, sn, seg_fname_mod, curve_ext)
    ip.plt.savefig(t_curve_filename, transparent=True)
    ip.plt.close()
    # Calculate the slope of the threshold curve
    t_curve_shift = t_curve[:1] + t_curve[:-1]
    slopes = [y2-y1 for y1, y2 in zip(t_curve_shift, t_curve)]
    # Find the threshold with the steepest negative slope and add  
    #     a standard value to that based on a visualization of threshold curves
    #     shifted horizontally to each other by anchoring to the steepest slope
#     print('slopes: ', slopes)
    slope_min = np.min(slopes)
    thresh_min = np.max(threshs[np.array(slopes) == slope_min])
#     print('Thresh_min: ', thresh_min)    
    int_thresh = thresh_min + int_thresh_normalized
    ###
    
    ### Fixed filter
#     int_thresh = int_thresh_normalized
    ###
    
    # Filter based on intensity
    filter_intensity = filter_distance.loc[(filter_distance.Intensity > int_thresh),:]
    # Remove large spots
    filter_area = filter_intensity.loc[(filter_intensity.Area < area_thresh),:]
    # Save the filtered table
    filtered_filename = '{}/{}{}{}'.format(out_dir, sn, seg_fname_mod, out_ext)
    filter_area.to_csv(filtered_filename)
    # Calculate the filtering stats
    tables = [spot_props, filter_distance, filter_intensity,filter_area]
    counts = [j.shape[0] for j in tables]
    count = [counts[0], counts[1], counts[2], counts[3], counts[3]]
    removed = [0, counts[0]-counts[1], counts[1]-counts[2], counts[2]-counts[3], 
               counts[0]-counts[3]]
    filter_value = ['n/a', dist_thresh, int_thresh, area_thresh, 'n/a']
    index = ['Initial','Large Distance Filter','Low Intensity Filter','Large Area Filter',
             'Totals']
    columns = ['Count','Removed','Filter value']
    zipped = list(zip(count, removed, filter_value))
    filter_stats = pd.DataFrame(zipped, columns=columns, index=index)
    # Save the filtering statistics
    filter_stats_filename = '{}/{}{}{}'.format(out_dir, sn, seg_fname_mod, stats_ext)
    filter_stats.to_csv(filter_stats_filename)
    
#     print('1\n')
    ###
    # Visualize filter steps
    ###
    raw_ext = args.raw_ext
    seg_ext = args.seg_ext
    flat_field_filename = args.flat_field_filename
    ff_index = args.ff_index
    figure_exts = args.fig_exts
    dims = args.dims
    clims = args.lims
    spot_channel_index = args.spot_channel_index
    
    # Load raw image and segmentation
    seg = np.load('{}/{}{}{}'.format(out_dir, sn, seg_fname_mod, seg_ext))
    raw = np.load('{}/{}{}'.format(out_dir, sn, seg_ext))[0:seg.shape[0],0:seg.shape[1],spot_channel_index] # the seg ignores the tiles, so we have to here as well 
    if flat_field_filename:
        flat_field = np.load(flat_field_filename) if not ff_index \
                 else np.load(flat_field_filename)[:,:,ff_index]
        raw_corrected = ifn.flat_field_correction(image=raw, flat_field=flat_field)
    else:
        raw_corrected = raw
    # Filter non-spots
    mask_bg = seg > 0
    im_filter_bg = raw_corrected * mask_bg 
    # Filter distance
    im_filter_dist = filter_raw(raw=im_filter_bg, seg=seg, props=filter_distance)    
#     print('2\n')
    # Filter Intensity
    im_filter_int = filter_raw(raw=im_filter_dist, seg=seg, props=filter_intensity)
#     print('3\n')
    # Filter Area
    im_filter_area = filter_raw(raw=im_filter_int, seg=seg, props=filter_area)    
#     print('4\n')
    # Create the figure
    ims = [im_filter_bg, im_filter_dist, im_filter_int, im_filter_area]
    fig, axs = ip.subplot_images(ims=ims, lims=clims, dims=dims)
    # Save filtering figure
    for ext in figure_exts:
        figure_filename = '{}/{}{}{}'.format(out_dir, sn, seg_fname_mod, ext)
        ip.plt.savefig(figure_filename, transparent=True, dpi=1000)
    ip.plt.close()
    return

    
if __name__ == '__main__':
    main()