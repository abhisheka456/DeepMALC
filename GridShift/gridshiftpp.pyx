import numpy as np
cimport numpy as np
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from datetime import datetime
from collections import defaultdict, Counter


cdef extern from "gridshiftpp.h":
    void generate_offsets_cy(int d, 
                             int base,
                             int * offsets)

    void grid_cluster(int n,
                  int d,
                  int base,
                  int iterations,
                  float bandwidth,
                  int * offsets,
                  float * X_shifted,
                  int * membership,
                  int * k_num)

cdef extern from "utils.h":
    void stitch_cy(int n,
                      int m, 
                      int threshold,
                      int n_clusters,
                      int * clusters,
                      int * updated_clusters)

cdef generate_offsets_np(d, 
                         base,
                         np.ndarray[np.int32_t, ndim=2, mode="c"] offsets):
    generate_offsets_cy(d, 
                        base,
                        <int *> np.PyArray_DATA(offsets))

cdef shift_np(n,
              d,
              base,
              iterations,
              bandwidth,
              np.ndarray[np.int32_t, ndim=2, mode="c"] offsets,
              np.ndarray[float, ndim=2, mode="c"] X_shifted,
              np.ndarray[np.int32_t, ndim=1, mode="c"] membership,
              np.ndarray[np.int32_t, ndim=1, mode="c"] k_num):
    grid_cluster(n,
             d,
             base,
             iterations,
             bandwidth,
             <int *> np.PyArray_DATA(offsets),
             <float *> np.PyArray_DATA(X_shifted),
             <int *> np.PyArray_DATA(membership),
             <int *> np.PyArray_DATA(k_num))

cdef stitch_np(n,
               m,
               threshold,
               n_clusters,
               np.ndarray[int, ndim=1, mode="c"] clusters,
               np.ndarray[int, ndim=1, mode="c"] updated_clusters):
    return stitch_cy(n,
                     m,
                     threshold,
                     n_clusters,
                     <int *> np.PyArray_DATA(clusters),
                     <int *> np.PyArray_DATA(updated_clusters))


class GridShiftPP:
    """
    Parameters
    ----------
    
    bandwidth: Radius for binning points. Points are assigned to the bin 
               corresponding to floor division by bandwidth

    threshold: Stop shifting if the L2 norm between iterations is less than
               threshold

    iterations: Maximum number of iterations to run

    """

    def __init__(self, bandwidth, threshold=0.0001, iterations=None):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.iterations = iterations

    def draw_box(self, X, x1, x2, y1, y2, cluster_vals=[0], mask_val=1):
      X_box = X[y1:y2, x1:x2, :]
      X_box = X_box.reshape((y2-y1) * (x2-x1), 3)
      result,__ = self.fit_predict(X_box, self.iterations)
      result = result.reshape(y2-y1, x2-x1)
      mask = np.full(X.shape[:2], -1, dtype=np.int32)
      mask[y1:y2, x1:x2] = np.where(np.isin(result, cluster_vals), mask_val, -1)
      obj = np.where(mask == mask_val)
      center = np.mean(obj, axis=1, dtype=np.int32)
      hist = Counter(tuple(map(tuple, np.floor(X[obj] / self.bandwidth))))
      return result, mask, center, hist

    def update_hist(self, X, mask, hist, mask_val=1): 
      obj = np.where(mask == mask_val)
      center = np.mean(obj, axis=1, dtype=np.int32)
      hist = Counter(tuple(map(tuple, np.floor(X[obj] / self.bandwidth))))

      return center, hist 

    def track(self, X, center, mask, hist, length, width, mask_val=1, adjust_threshold=0.75):
      l, w, _ = X.shape
      iteration = 0

      while not self.iterations or iteration < self.iterations:
        iteration += 1

        mask = np.full((l, w), -1, dtype=np.int32)
        min_y = l
        max_y = 0
        min_x = w 
        max_x = 0

        # track_np(l, w, length, width, self.bandwidth, X, hist, mask)
        
        for i in range(length + 1):
          for j in range(width + 1):
            y = center[0] + i - int(length/2)
            x = center[1] + j - int(width/2)
            if x < 0 or x >= w or y < 0 or y >= l:
              continue
            bin_ = tuple(np.floor(X[y][x] / self.bandwidth))
            if hist[bin_] > 0:
              mask[y][x] = mask_val
              min_y = min(min_y, y)
              max_y = max(max_y, y)
              min_x = min(min_x, x)
              max_x = max(max_x, x)

        new_center, hist = self.update_hist(X, mask, hist, mask_val=mask_val)
        if max_y - min_y < adjust_threshold * length:
          length = int(max_y - min_y)
        if max_x - min_x < adjust_threshold * width:
          width = int(max_x - min_x)

        if np.linalg.norm(np.subtract(center, new_center)) <= self.threshold:
            break

        center = new_center

      return center, mask, hist, length, width

    def fit_predict(self, X, return_modes=False):
        """
        Determines the clusters in either `iterations` or when the L2 
        norm of consecutive iterations is less than `threshold`, whichever 
        comes first.
        Each shift has two steps: First, points are binned based on floor 
        division by bandwidth. Second, each bin is shifted to the 
        weighted mean of its 3**d neighbors. 
        Lastly, points that are in the same bin are clustered together.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
           Euclidean space

        Returns
        ----------
        (n, ) cluster labels
        """
        
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        X_shifted = np.copy(X)

        membership = np.full(n, -1, dtype=np.int32)

        iteration = 0
        base =  3
        offsets = np.full((base**d, d), -1, dtype=np.int32)
        generate_offsets_np(d, base, offsets)
        k = np.full((1,), -1, dtype=np.int32);

        shift_np(n, d, base, self.iterations, self.bandwidth, offsets, X_shifted, membership, k)

        centres = X_shifted[0:np.ndarray.item(k),:];

        return membership, centres
