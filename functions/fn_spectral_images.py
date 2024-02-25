# Fuctions for dealing with HiPRFISH spectral images from zeiss i880

# =============================================================================
# Imports
# =============================================================================

from skimage.registration import phase_cross_correlation
import numpy as np
from skimage.measure import regionprops

# =============================================================================
# Functions
# =============================================================================


def _shift_images(image_stack, shift_vectors, max_shift):
    image_registered = [np.zeros(image.shape) for image in image_stack]
    # shift_filter_mask = [
    #         np.full(
    #                 (image.shape[0], image.shape[1]),
    #                 False, dtype = bool
    #                 )
    #         for image in image_stack
    #         ]
    # ims_arr = np.array([ims.shape for ims in image_stack])
    # ims_shft = []
    # for ims, shft in zip(ims_arr, np.array(shift_vectors)):
    #     ims_shft.append(np.diff(ims+shft))
    # ims_max = np.max(image_shapes, axis=0)
    # shfts_max = np.max(np.array(shift_vectors), axis=0)
    for i in range(len(image_stack)):
        image_shape = image_stack[i].shape
        shift_row = int(shift_vectors[i][0])
        shift_col = int(shift_vectors[i][1])
        print(i, shift_row, shift_col)
        if np.abs(shift_row) > max_shift:
            shift_row = 0
        if np.abs(shift_col) > max_shift:
            shift_col = 0
        original_row_min = int(np.maximum(0, shift_row))
        original_row_max = int(image_shape[0] + np.minimum(0, shift_row))
        original_col_min = int(np.maximum(0, shift_col))
        original_col_max = int(image_shape[1] + np.minimum(0, shift_col))
        registered_row_min = int(-np.minimum(0, shift_row))
        registered_row_max = int(image_shape[0] - np.maximum(0, shift_row))
        registered_col_min = int(-np.minimum(0, shift_col))
        registered_col_max = int(image_shape[1] - np.maximum(0, shift_col))
        image_registered[i][original_row_min: original_row_max, original_col_min: original_col_max, :] = image_stack[i][registered_row_min: registered_row_max, registered_col_min: registered_col_max, :]
        # shift_filter_mask[i][original_row_min: original_row_max, original_col_min: original_col_max] = True
    return image_registered


def _get_shift_vectors(image_sum):
    # Find shift vectors
    shift_vectors = [
            phase_cross_correlation(
                    np.log(image_sum[0]+1), np.log(image_sum[i]+1)
                    )[0]
            for i in range(1,len(image_sum))
            ]
    shift_vectors.insert(0, np.asarray([0.0,0.0]))
    return shift_vectors


def _size_images(image_list):
    print([im.shape for im in image_list])
    max_r = np.max([im.shape[0] for im in image_list])
    max_c = np.max([im.shape[1] for im in image_list])
    image_stack = []
    for im in image_list:
        im_resz = np.zeros((max_r, max_c, im.shape[2]))
        im_resz[:im.shape[0], :im.shape[1], :im.shape[2]] = im
        image_stack.append(im_resz)
    return image_stack


def register_shifts(image_list, max_shift=20):
    # Make all images the same size in case stitching made variations
    image_stack = _size_images(image_list)
    # Get projection for each channel
    image_sum = [np.sum(image, axis = 2) for image in image_stack]
    # Get the shifts
    shift_vectors = _get_shift_vectors(image_sum)
    print(shift_vectors)
    # Shift the images
    image_registered = _shift_images(image_stack, shift_vectors, max_shift)
    shp = np.min([ims.shape[:2] for ims in image_registered], axis=0)
    print(shp)
    image_registered_trimmed = [im[:shp[0],:shp[1],:] for im in image_registered]
    # Get outputs
    image_registered_cat = np.concatenate(image_registered_trimmed, axis = 2)
    image_registered_max = np.max(image_registered_cat, axis = 2)
    image_registered_sum = np.sum(image_registered_cat, axis = 2)
    return(image_registered_cat, image_registered_max, image_registered_sum, shift_vectors)


def get_cell_average_spectra(seg, raw):
    n_cells = np.unique(seg).shape[0] - 1
    avgint = np.empty((n_cells, raw.shape[2]))
    for k in range(0, raw.shape[2]):
        cells = regionprops(seg, intensity_image = raw[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    return avgint


def plot_cell_spectra(ax, arr_spec, kwargs):
    x = np.arange(arr_spec.shape[1])
    X = np.ones(arr_spec.shape) * x
    ax.plot(X.T, arr_spec.T, **kwargs)
    return(ax)


def recolor_image(im, dict_label_bbox, dict_label_alt, threeD=0):
    shape = im.shape + (threeD,) if threeD else im.shape
    im_alt = np.zeros(shape)
    for label, bbox in dict_label_bbox.items():
        if isinstance(bbox, str):
            bbox = eval(bbox)
        box = (im[bbox[0]:bbox[2],bbox[1]:bbox[3]] == label)*1
        if label == 1: print(np.unique(box))
        box = box[...,None] if threeD else box
        alt = dict_label_alt[label]
        alt = eval(alt) if isinstance(alt, str) else alt
        box_alt = (box * alt) + im_alt[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        # if label==1: print(box_alt.shape, np.unique(box_alt))
        im_alt[bbox[0]:bbox[2],bbox[1]:bbox[3]] = box_alt
    return(im_alt)


def plot_nn_dists(ax, nn_dists, kwargs):
    nn_dists_sort = np.sort(nn_dists)
    x = np.arange(nn_dists_sort.shape[0]) + 1
    ax.scatter(x, nn_dists_sort, **kwargs)
    return ax
