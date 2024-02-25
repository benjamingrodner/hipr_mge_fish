# from IPython.display import HTML, display
# from IPython.display import Image as _Image
import glob
import re
import numpy as np
import pandas as pd
from skimage.util import pad
from numba import njit
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.measure import regionprops


def get_sample_names(data_dir='../data/images', sample_glob='*', ext='.czi', re_glob=''):
    filename_glob = '{}/{}{}'.format(data_dir, sample_glob, ext)
    filenames = glob.glob(filename_glob)
    sample_filenames = [re.sub(data_dir + '/', '', s) for s in filenames]
    return list(set([re.sub(re_glob + ext, '', f) for f in sample_filenames]))



def get_filename_keys(sample_name, filename_factors, sep='_'):
    keys = []
    for factor in filename_factors:
        match = re.search(r'(?<=' + sep + factor + sep + ')[0-9A-Za-z.]+',
                          sample_name)
        if match:
            level = match.group()
            keys.append(level)
    return keys


def _src_from_data(data):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    img_obj = _Image(data=data)
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return 'data:{mimetype};base64,{b64value}'.format(mimetype=mimetype, b64value=b64value)


def gallery(images, captions='auto', row_height='auto'):
    """Shows a set of images in a gallery that flexes with the width of the notebook.

    Parameters
    ----------
    images: list of str or bytes
        URLs or bytes of images to display

    row_height: str
        CSS height value to assign to all images. Set to 'auto' by default to show images
        with their native dimensions. Set to a value like '250px' to make all rows
        in the gallery equal height.
    """
    figures = []
    if captions == 'auto':
        captions = ['auto' for i in images]
    for image, cap in zip(images, captions):
        if isinstance(image, bytes):
            src = _src_from_data(image)
            caption = ''
        else:
            src = image
            cap_str = image if cap == 'auto' else cap
            caption = '<figcaption style="font-size: 0.6em">{}</figcaption>'.format(cap_str)
        figures.append('''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        '''.format(src=src, row_height=row_height, caption=caption))
    return HTML(data='''
        <div style='display: flex; flex-flow: row wrap; text-align: center;''>
        {fig}
        </div>
    '''.format(fig=''.join(figures)))


@njit
def _get_spot_max_cell_ids_numba(cell_seg, spot_coords, ids, radius, downsample):
    id_array = np.empty((spot_coords.shape[0], 3))
    for i in range(spot_coords.shape[0]):
        xi = spot_coords[i,0]
        yi = spot_coords[i,1]
        # Get the cell id for the centroid of the spot
        cell_id = cell_seg[yi,xi]
        dist = 0
        # If the spot is not in a cell, check nearby
        if cell_id == 0:
            dist = 1e5
            # Check in a circle around the centroid
            for xr in np.arange(xi - radius, xi + radius + downsample, downsample):
                for yr in np.arange(yi - radius, yi + radius + downsample, downsample):
                    if (xi - xr)**2 + (yi - yr)**2 <= radius**2:
                        nearby_cell = cell_seg[yr,xr]
                        if not nearby_cell == 0:
                            dist_new = ((xr-xi)**2 + (yr-yi)**2)**(1/2)
                            if dist_new < dist:
                                dist = dist_new
                                cell_id = nearby_cell
        # Add the end cell id and distance to the temp dataframe
        id_array[i,:] = [int(ids[i]), int(cell_id), dist]
    return(id_array)


def _prepare_numba_inputs(seg, props, search_radius):
    cs_pad = pad(seg, search_radius, mode='edge')
    coords = props.loc[:,['X','Y']].copy()
    coords = coords.values.astype('int64')
    coords = coords + search_radius
    max_ids = props.loc[:,['ID']].copy()
    max_ids = max_ids.values.astype('int64').flatten()
    return(cs_pad, coords, max_ids)


def spots_to_cell_id(cell_seg, spot_props, search_radius=10, downsample=1):
    seg_pad, coords, ids = _prepare_numba_inputs(cell_seg, spot_props, search_radius)
    spot_cell_id_array = _get_spot_max_cell_ids_numba(seg_pad, coords, ids,
                                                          search_radius, downsample)
    spot_cell_id_df = pd.DataFrame(spot_cell_id_array, columns = ['ID','cell_id','dist'])
    return spot_props.merge(spot_cell_id_df, on='ID', how='left')


def _group(kv_pairs, k_index):
    list_dict = defaultdict(list)
    for kv in kv_pairs:
        group = kv[0][k_index]
        list_dict[group].append(kv)
    return list_dict


def get_nested_dict(key_list, values_list, groupby_key_indices):
    kv_zip = zip(key_list, values_list)
    n_nests = len(groupby_key_indices)
    nest_dict = _group(kv_zip, groupby_key_indices[0])
    if n_nests > 1:
        for k1, v1 in nest_dict.items():
            nest_dict[k1] = _group(v1, groupby_key_indices[1])
            if n_nests > 2:
                for k2, v2 in nest_dict[k1].items():
                    nest_dict[k1][k2] = _group(v2, groupby_key_indices[2])
                    if n_nests > 3:
                        for k3, v3 in nest_dict[k1][k2].items():
                            nest_dict[k1][k2][k3] = _group(v3, groupby_key_indices[3])
    return nest_dict


def base_10_to_base_2(num):
    i = 0
    while 2**i < num:
        i +=1
    left = num
    converted = ''
    for j in range((i-1),-1,-1):
        power = 2**(j)
        if power <= left:
            converted += '1'
            left -= power
        else:
            converted += '0'
    return int(converted)


@njit
def spot_filter_mask(seg, retained_ids):
    mask = np.zeros(seg.shape)
    for f in retained_ids:
        _m = seg == f
        mask += _m
    return mask


def flat_field_correction(image, flat_field):
    corrected = image/(flat_field + 1e-15)
    maxs = [np.max(i) for i in [corrected, image]]
    corrected = corrected * maxs[1] / maxs[0]
    return corrected


def measure_regionprops(seg, raw):
    regions = regionprops(seg, intensity_image = raw)
    region_properties = pd.DataFrame(columns = ['ID', 'X', 'Y', 'Area', 'MeanIntensity','MaxIntensity'])
    for j in range(len(regions)):
        region_properties.loc[j, 'ID'] = regions[j].label
        region_properties.loc[j, 'X'] = regions[j].centroid[1]
        region_properties.loc[j, 'Y'] = regions[j].centroid[0]
        region_properties.loc[j, 'Area'] = regions[j].area
        region_properties.loc[j, 'MeanIntensity'] = regions[j].mean_intensity
        region_properties.loc[j, 'MaxIntensity'] = regions[j].max_intensity
#         region_properties.loc[j, 'major_axis_length'] = regions[j].major_axis_length
#         region_properties.loc[j, 'minor_axis_length'] = regions[j].minor_axis_length
    return region_properties
