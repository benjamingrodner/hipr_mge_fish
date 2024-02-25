###############################################################################
# Face morpher functions
# Adapted from https://github.com/alyssaq/face_morpher
###############################################################################

import numpy as np
import scipy.spatial as spatial

def bilinear_interpolate(img, coords):
  """ Interpolates over every image channel
  http://en.wikipedia.org/wiki/Bilinear_interpolation
  :param img: max 3 channel image
  :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
  :returns: array of interpolated pixels with same shape as coords
  """
  int_coords = np.int32(coords)
  x0, y0 = int_coords
  dx, dy = coords - int_coords

  # 4 Neighour pixels
  q11 = img[y0, x0]
  q21 = img[y0, x0+1]
  q12 = img[y0+1, x0]
  q22 = img[y0+1, x0+1]

  btm = q21.T * dx + q11.T * (1 - dx)
  top = q22.T * dx + q12.T * (1 - dx)
  inter_pixel = top * dy + btm * (1 - dy)
  # print('inter_pixel',inter_pixel[:10])
  return inter_pixel.T

def nearest_interpolate(img, coords):
  """ Interpolates over every image channel
  http://en.wikipedia.org/wiki/Bilinear_interpolation
  :param img: max 3 channel image
  :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
  :returns: array of interpolated pixels with same shape as coords
  """
  int_coords = np.int32(coords)
  x0, y0 = int_coords
  dx, dy = coords - int_coords

  # 4 Neighour pixels
  q11 = img[y0, x0]
  q21 = img[y0, x0+1]
  q12 = img[y0+1, x0]
  q22 = img[y0+1, x0+1]

  btm = q21.T * dx + q11.T * (1 - dx)
  top = q22.T * dx + q12.T * (1 - dx)
  inter_pixel = top * dy + btm * (1 - dy)

  return inter_pixel.T

def grid_coordinates(points):
  """ x,y grid coordinates within the ROI of supplied points
  :param points: points to generate grid coordinates
  :returns: array of (x, y) coordinates
  """
  xmin = np.min(points[:, 0])
  xmax = np.max(points[:, 0]) + 1
  ymin = np.min(points[:, 1])
  ymax = np.max(points[:, 1]) + 1
  return np.asarray([(x, y) for y in range(ymin, ymax)
                     for x in range(xmin, xmax)], np.uint32)

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay,
                 interpolation):
  """
  Warp each triangle from the src_image only within the
  ROI of the destination image (points in dst_points).
  """
  roi_coords = grid_coordinates(dst_points)
  # indices to vertices. -1 if pixel is not in any triangle
  roi_tri_indices = delaunay.find_simplex(roi_coords)

  for i, simplex_index in enumerate(range(len(delaunay.simplices))):
    coords = roi_coords[roi_tri_indices == simplex_index]
    num_coords = len(coords)
    out_coords = np.dot(tri_affines[simplex_index],
                        np.vstack((coords.T, np.ones(num_coords))))
    x, y = coords.T
    # print(x,y)
    if interpolation == 'bilinear':
        bi = bilinear_interpolate(src_img, out_coords)
    elif interpolation == 'nearest':
        x0, y0 = np.rint(out_coords).astype(np.int32)
        bi = src_img[y0, x0]
    result_img[y, x] = bi

  return None

def triangular_affine_matrices(vertices, src_points, dest_points):
  """
  Calculate the affine transformation matrix for each
  triangle (x,y) vertex from dest_points to src_points
  :param vertices: array of triplet indices to corners of triangle
  :param src_points: array of [x, y] points to landmarks for source image
  :param dest_points: array of [x, y] points to landmarks for destination image
  :returns: 2 x 3 affine matrix transformation for a triangle
  """
  ones = [1, 1, 1]
  for i, tri_indices in enumerate(vertices):

    src_tri = np.vstack((src_points[tri_indices, :].T, ones))
    dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
    mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
    yield mat

def warp_image(src_img, src_points, dest_points, dest_shape,
                dtype=np.uint8, interpolation='bilinear'):
  # Resultant image will not have an alpha channel
  if len(src_img.shape) == 2:
    num_chans = 1 if len(src_img.shape) == 2 else src_img.shape[2]
    rows, cols = dest_shape
    result_img = np.zeros((rows, cols), dtype)
  else:
    src_img = src_img[:, :, :3]
    num_chans = src_img.shape[2]
    rows, cols = dest_shape[:2]
    result_img = np.zeros((rows,cols, num_chans), dtype)

  delaunay = spatial.Delaunay(dest_points)
  tri_affines = np.asarray(list(triangular_affine_matrices(
    delaunay.simplices, src_points, dest_points)))

  process_warp(src_img, result_img, tri_affines, dest_points, delaunay, interpolation)
  return result_img

def weighted_average_points(start_points, end_points, percent=0.5):
  """ Weighted average of two sets of supplied points
  :param start_points: *m* x 2 array of start face points.
  :param end_points: *m* x 2 array of end face points.
  :param percent: [0, 1] percentage weight on start_points
  :returns: *m* x 2 array of weighted average points
  """
  if percent <= 0:
    return end_points
  elif percent >= 1:
    return start_points
  else:
    return np.asarray(start_points*percent + end_points*(1-percent), np.int32)
