from image import Image
import os
import sys
import glob
import copy
import argparse
import numpy as np
from sklearn import svm
from skimage.filters import threshold_otsu, threshold_sauvola, threshold_niblack
from scipy import ndimage as ndi
import skimage
from skimage.feature import peak_local_max
from skimage import morphology, segmentation
from skimage import restoration
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from skimage import color
import pandas as pd
import re
from neighbor2d import line_profile_2d_v2
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_opening
import javabridge
import bioformats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.segmentation import find_boundaries
from skimage.restoration import denoise_nl_means


class Segmentation(Image):
    '''
    Implementation of the LNE segmentation algorithm from HiPR-FISH
    Inherits args from Images:
        illumination_names, data_dir='../data/images', filenames_glob='*', ext='.czi'
    Args:
        window - half width of LNE measurement in pixels
    '''
    def __init__(self, sample_name, illumination_names,
                 data_dir='../data/images', ext='.czi',
                 out_dir='../image_processing',
                 split_files=False, fname_mod='_seg', fig_ext='.png',
                 window=5, n_clusters=2, top_n_clust=1, bg_smoothing=5, bg_filter_channel=False,
                 lne_smoothing=0, bg_filter=True,
                 lne_log=False, bg_log=False, small_objects=9,
                 flat_field_filename=False,
                 flat_field_index=0):
        super().__init__(sample_name, illumination_names, data_dir, ext, out_dir,
                         split_files, flat_field_filename, flat_field_index)
        self.window = window
        self.n_clusters = n_clusters
        self.top_n_clust = top_n_clust
        self.bg_smoothing = bg_smoothing
        self.bg_filter_channel = bg_filter_channel
        self.lne_smoothing = lne_smoothing
        self.bg_filter = bg_filter
        self.bg_log = bg_log
        self.lne_log = lne_log
        self.small_objects = small_objects
        self.fname_mod = fname_mod

        self.basename = self.out_dir + '/' + sample_name + fname_mod
        self.seg_filename = self.basename + '.npy'
        self.seg_process_filename = self.basename + '_process' + fig_ext
        self.region_props_filename = self.basename + '_props.csv'

    def _split_large_image(self, size=2000, overlap=5):
        output_folder = self.basename + '_split'
        if not os.path.exists(output_folder): os.mkdir(output_folder)
        n_rows = np.floor(self.raw_2D.shape[0]/size).astype(int)
        r_rows = (self.raw_2D.shape[0]/size - n_rows) * self.raw_2D.shape[0]
        n_cols = np.floor(self.raw_2D.shape[1]/size).astype(int)
        r_cols = (self.raw_2D.shape[1]/size - n_cols) * self.raw_2D.shape[1]
        print('Split image into {} x {} images for lne processing'.format(n_rows+1, n_cols+1))        
        self.split_keys = []
        self.split_filenames = []
        for r in range(n_rows+1):
            if r == 0:
                y = [0, size + overlap]
            elif r < n_rows-1:
                y = [size * r - overlap, size * (r + 1) + overlap]
            else:
                if r_rows <= overlap:
                    if r == n_rows-1:
                        y = [size * r - overlap, self.raw_2D.shape[0]]
                    else:
                        pass
                else:
                    if r == n_rows-1:
                        y = [size * r - overlap, size * (r + 1) + overlap]
                    else:
                        y = [size * r - overlap, self.raw_2D.shape[0]]
            for c in range(n_cols + 1):
                if c == 0:
                    x = [0, size + overlap]
                elif c < n_cols-1:
                    x = [size * c - overlap, size * (c + 1) + overlap]
                else:
                    if r_cols <= overlap:
                        if c == n_cols-1:
                            x = [size * c - overlap, self.raw_2D.shape[1]]
                        else:
                            pass
                    else:
                        if c == n_cols-1:
                            x = [size * c - overlap, size * (c + 1) + overlap]
                        else:
                            x = [size * c - overlap, self.raw_2D.shape[1]]
                print('Split info',r,c,y,x)
                subset = self.raw_2D[y[0]:y[1],x[0]:x[1]]
                output_fn = output_folder + '/row_' + str(r) + '_col_' + str(c) + '.npy'
                np.save(output_fn, subset)
                self.split_filenames.append(output_fn)
                self.split_keys.append([r,c])


    def _large_file_lne(self):
        self._split_large_image()
        for fn in self.split_filenames:
            output_fn = re.sub('.npy', '_seg.npy', fn)
#             if not os.path.exists(output_fn):
            raw_2D = np.load(fn)
            self._apply_lne(raw_2D=raw_2D)
            self._get_rough_segmentation()
            self._get_background_filter(raw_2D=raw_2D)
            self._watershed_segmentation()
            np.save(output_fn, self._seg)
            self.save_seg(self, ext='.png',
                         raw_illumination=False,
                         dims=(1, 1), clims=(0,0.05), cmap='inferno', zoom_coords=[],
                         save=True, split_filename=fn)    
            print('Processed: ', os.path.split(output_fn)[1])
        np.save(self.seg_filename, np.array([]))


    def _apply_lne(self, raw_2D=np.array([])):
        raw_2D = self.raw_2D if raw_2D.shape[0] == 0 else raw_2D
        image = np.log10(raw_2D + 1e-15) if self.lne_log else raw_2D
        if self.lne_smoothing:
#             image_smooth = denoise_nl_means(image, patch_size=7, patch_distance=11, h=0.1, sigma=0.0)
            image_smooth = ndi.gaussian_filter(image, sigma=self.lne_smoothing, order=0)
        else:
            image_smooth = image
        self.seg_input = image_smooth
        image_padded = skimage.util.pad(image_smooth, self.window, mode = 'edge')
        print(image_smooth.shape)
        image_lp = line_profile_2d_v2(image_padded.astype(np.float64), self.window*2 + 1, 9)
        image_lp = np.nan_to_num(image_lp)
        image_lp_min = np.min(image_lp, axis = 3)
        image_lp_max = np.max(image_lp, axis = 3)
        image_lp_max = image_lp_max - image_lp_min
        image_lp = image_lp - image_lp_min[:,:,:,None]
        image_lp_max = image_lp_max + 1e-5
        image_lp_rel_norm = image_lp/image_lp_max[:,:,:,None]
        image_lp_rnc = image_lp_rel_norm[:,:,:,self.window]
        image_lprns = np.average(image_lp_rnc, axis = 2)
        image_lprn_lq = np.percentile(image_lp_rnc, 25, axis = 2)
        image_lprn_uq = np.percentile(image_lp_rnc, 75, axis = 2)
        image_lprn_qcv = np.zeros(image_lprn_uq.shape)
        image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq)/(image_lprn_uq + image_lprn_lq + 1e-8)
        image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
        self._lne = image_lprns*(1-image_lprn_qcv)


    def _get_rough_segmentation(self):
        n_clust = 2
        image_final_clustered = MiniBatchKMeans(n_clusters = n_clust, batch_size = 100000,
                                                random_state = 42)
        image_final_clustered = image_final_clustered.fit_predict(self._lne.reshape(-1,1))
        image_final_clustered = image_final_clustered.reshape(self._lne.shape)
        list_i0 = []
        for i in range(n_clust):
            image_ = self._lne*(image_final_clustered == i)
            i0 = np.average(image_[image_ > 0])
            list_i0.append(i0)
        intensity_rough_seg = image_final_clustered == np.argmax(list_i0)
        intensity_rough_seg = binary_opening(intensity_rough_seg)
        intensity_rough_seg = skimage.morphology.remove_small_objects(intensity_rough_seg, 1)
        self._intensity_rough_seg = binary_fill_holes(intensity_rough_seg)


    def _get_background_filter(self, raw_2D=np.array([])):
        raw_2D = self.raw_2D if raw_2D.shape[0] == 0 else raw_2D
        image = self._get_2D(self.bg_filter_channel) if self.bg_filter_channel else raw_2D
        if self.bg_log:
            image = np.log10(image + 1e-15)
            image = image - np.min(image)
        if self.bg_smoothing:
            image_smooth = ndi.gaussian_filter(image, sigma=self.bg_smoothing, order=0)
        else:
            image_smooth = raw_2D
        if self.bg_filter:
            image_bkg_filter = MiniBatchKMeans(n_clusters = self.n_clusters,
                                               batch_size = 100000, random_state = 42)
            shape_ = image_smooth.reshape(np.prod(image_smooth.shape), 1)
            image_bkg_filter = image_bkg_filter.fit_predict(shape_)
            image_bkg_filter = image_bkg_filter.reshape(image_smooth.shape)
            i_list = []
            for n in range(self.n_clusters):
                image_ = image*(image_bkg_filter == n)
                i_n = np.average(image_[image_ > 0])
                i_list.append(i_n)
            i_list = np.argsort(i_list)[::-1]
            self._image_bkg_filter_mask = np.zeros(image_bkg_filter.shape, dtype=bool)
            for tn in range(self.top_n_clust):
                self._image_bkg_filter_mask += image_bkg_filter == i_list[tn]
        else:
            self._image_bkg_filter_mask = np.ones(raw_2D.shape)


    def _watershed_segmentation(self, lne_watershed_mask=False):
#         image_smooth = self.raw_2D.copy()
#         print(image_smooth.shape)
#         cutoff = 0.05
#         image_smooth[image_smooth > cutoff] = cutoff
#         watershed_input = ndi.gaussian_filter(self.raw_2D, sigma=10, order=0)
#         self.watershed_input = self.seg_input*self._image_bkg_filter_mask
        self.watershed_input = self._lne*self._image_bkg_filter_mask
        seeds = self._intensity_rough_seg*self._image_bkg_filter_mask*1
        image_watershed_mask = seeds if lne_watershed_mask else self._image_bkg_filter_mask
        image_watershed_seeds = skimage.morphology.label(seeds)

        image_seg = skimage.segmentation.watershed(-self.watershed_input,
                                                   image_watershed_seeds,
                                                   mask = image_watershed_mask)
        self.image_watershed_seeds = image_watershed_seeds
        self._seg = skimage.morphology.remove_small_objects(image_seg, self.small_objects)


    def _show_seg_on_raw(self, ax, raw_illumination, clims, cmap, zoom_coords, split_filename=False):
        cmap_temp = copy.copy(plt.cm.get_cmap('gray'))
        cmap_temp.set_bad(alpha = 0)
        cmap_temp.set_over((0,1,0), 1.0)
        seg = np.load(self.seg_filename) if not split_filename else self._seg
        im_line = find_boundaries(seg, mode = 'outer')
        im_line = im_line.astype(float)
        im_line[im_line == 0] = np.nan
        if not split_filename:
            raw_illumination = self.illumination if not raw_illumination else raw_illumination
            raw_illumination = [raw_illumination] if not isinstance(raw_illumination, list) else raw_illumination
            channel_index = [self.illumination_names.index(il) for il in raw_illumination]
            raw = self.raw_stack[:, :, channel_index]
            raw = np.sum(raw, axis=2) if len(raw.shape) > 2 else raw
            ## Added for tile scan issues
            # raw = raw[:2000,:2000]
        else:
            raw = np.load(split_filename)
        extent = 0, raw.shape[1], raw.shape[0], 0
        if self.flat_field_filename:
            raw = self._flat_field_correction(image=raw)
        llim = np.min(raw) if clims[0]  == 'min' else float(clims[0])
        ulim = np.max(raw) if clims[1]  == 'max' else float(clims[1])
        ax.imshow(raw, cmap = cmap, extent=extent, clim = (llim, ulim))
        ax.imshow(im_line, cmap = cmap_temp, norm=colors.Normalize(vmin=0, vmax=0.1), extent = extent)
        zc = extent if len(zoom_coords)==0 else zoom_coords
        ax.set_xlim(zc[0], zc[1])
        ax.set_ylim(zc[3], zc[2])


    def _plot_seg_process(self, zoom_coords=(), clims=('min','max'), dims=(10, 5), cmap='inferno'):
        xmax = self.raw_2D.shape[1] - 1
        ymax = self.raw_2D.shape[0] - 1
        fig = plt.figure(frameon=False)
        ims = (self.raw_2D, self._lne, self._intensity_rough_seg,
               self._image_bkg_filter_mask, self.watershed_input, self._seg)
        titles = ('image', 'lne', 'intensity_rough_seg',
                  'image_bkg_filter_mask','watershed input','image_seg')
        overlay = (0,0,0,0,0,1)
        continuous_cmap = (1, 1, 0, 0, 1, 0)
        for i, (im, title, overl) in enumerate(zip(ims, titles, overlay)):
            ax = fig.add_subplot(2, 3, i+1)
            if not overl:
                if continuous_cmap:
                    self._show_intensity_image(im, ax, clims, cmap, zoom_coords)
                else:
                    ax.imshow(im)
                    x1, x2, y1, y2 = (0, xmax, 0, ymax) if len(zoom_coords)==0 else zoom_coords
                    ax.set_xlim(x1, x2)
                    ax.set_ylim(y1, y2)
            else:
                self._show_seg_on_raw(ax, raw_illumination=False, clims=clims, cmap=cmap,
                                      zoom_coords=zoom_coords)
            ax.set_title(title)
            fig.set_size_inches(dims[0], dims[1])
        return(fig, ax)


    def segment(self, lne_watershed_mask):
        if (self.raw_2D.shape[0] > 2000) | (self.raw_2D.shape[1] > 2000):
            print('Large image, splitting...')
            self._large_file_lne()
        else:
            print('Segmenting...')
            self._apply_lne()
            self._get_rough_segmentation()
            self._get_background_filter()
            self._watershed_segmentation(lne_watershed_mask=lne_watershed_mask)
            np.save(self.seg_filename, self._seg)


    def save_seg(self, fname_mod='_seg', ext='.png',
                 raw_illumination=False,
                 dims=(5, 5), clims=(0,1), cmap='inferno', zoom_coords=[], save=True, split_filename=False):
        fig, ax = self._image_figure(dims)
        self._show_seg_on_raw(ax, raw_illumination, clims, cmap, zoom_coords, split_filename=split_filename)
        if save:
            output_filename = self.basename + fname_mod + ext if not split_filename else re.sub('.npy','_seg.png',split_filename)
            plt.savefig(output_filename, dpi=self._seg.shape[0]//np.min(dims))
            plt.close()
            self.seg_plot_filename = output_filename
        else:
            plt.show()
            plt.close()


    def save_seg_process(self, zoom_coords=(), clims=(0,1), dims=(10, 5),
                         cmap='inferno', save=True):
        fig, ax = self._plot_seg_process(zoom_coords, clims, dims, cmap)
        if save:
            plt.savefig(self.seg_process_filename, dpi=self.raw_2D.shape[0]//np.min(dims))
        else:
            plt.show()
        plt.close()


    def measure_regionprops(self, fname_mod='_props'):
        regions = skimage.measure.regionprops(self._seg, intensity_image = self.raw_2D)
        region_properties = pd.DataFrame(columns = ['ID', 'X', 'Y', 'Area', 'Intensity', 'ParentCellID'])
        for j in range(len(regions)):
            region_properties.loc[j, 'ID'] = regions[j].label
            region_properties.loc[j, 'X'] = regions[j].centroid[1]
            region_properties.loc[j, 'Y'] = regions[j].centroid[0]
            region_properties.loc[j, 'Area'] = regions[j].area
            region_properties.loc[j, 'Intensity'] = regions[j].mean_intensity
            region_properties.loc[j, 'major_axis_length'] = regions[j].major_axis_length
            region_properties.loc[j, 'minor_axis_length'] = regions[j].minor_axis_length
        region_properties.to_csv(self.region_props_filename)
