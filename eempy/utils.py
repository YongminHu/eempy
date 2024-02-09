"""
Useful functions for EEM processing
"""


import numpy as np
import itertools
from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta


def dichotomy_search(nums, target):
    start = 0
    end = len(nums) - 1
    if target < min(nums):
        return np.where(nums == min(nums))[0][0]
    if target > max(nums):
        return np.where(nums == max(nums))[0][0]
    while start <= end:
        mid = (start + end) // 2
        fdiff = nums[mid] - target
        bdiff = nums[mid - 1] - target
        if fdiff * bdiff <= 0:
            if abs(fdiff) < abs(bdiff):
                return mid
            if abs(bdiff) <= abs(fdiff):
                return mid - 1
        elif nums[mid] < target:
            start = mid + 1
        else:
            end = mid - 1


def euclidean_dist_for_tuple(t1, t2):
    dist = 0
    for x1, x2 in zip(t1, t2):
        dist += (x1 - x2) ** 2
    return dist ** 0.5


def matrix_dtype_to_uint8(m):
    m_r = np.copy(m)
    m_r[m_r < 0] = 0
    m_scaled = np.interp(m_r, (0, m_r.max()), (0, 255))
    m_scaled = m_scaled.astype(np.uint8)
    return m_scaled


def datetime_to_str(datetime_list, output=False, filename='timestamp.txt'):
    tsstr = [datetime_list[i].strftime("%Y-%m-%d-%H-%M") for i in range(len(datetime_list))]
    if output:
        file = open(filename, 'w')
        for fp in tsstr:
            file.write(str(fp))
            file.write('\n')
        file.close()
    return tsstr


def dynamic_time_warping(x, y):
    # Create a cost matrix with initial values set to infinity
    cost_matrix = np.ones((len(x), len(y))) * np.inf
    # Initialize the first cell of the cost matrix to the Euclidean distance between the first elements
    cost_matrix[0, 0] = euclidean([x[0]], [y[0]])
    # Calculate the cumulative cost matrix
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            cost_matrix[i, j] = euclidean([x[i]], [y[j]]) + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1],
                                                                      cost_matrix[i - 1, j - 1])
    # Trace back the optimal path
    i, j = len(x) - 1, len(y) - 1
    path = [(i, j)]
    while i > 0 and j > 0:
        if cost_matrix[i - 1, j] <= cost_matrix[i, j - 1] and cost_matrix[i - 1, j] <= cost_matrix[i - 1, j - 1]:
            i -= 1
        elif cost_matrix[i, j - 1] <= cost_matrix[i - 1, j] and cost_matrix[i, j - 1] <= cost_matrix[i - 1, j - 1]:
            j -= 1
        else:
            i -= 1
            j -= 1
        path.append((i, j))
    path.reverse()
    # Extract the aligned arrays based on the optimal path
    aligned_x = [x[i] for i, _ in path]
    aligned_y = [y[j] for _, j in path]
    return aligned_x, aligned_y


def flip_legend_order(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def get_indices_smallest_to_largest(my_list):
    indexed_list = [(value, index) for index, value in enumerate(my_list)]
    sorted_list = sorted(indexed_list, key=lambda x: x[0])
    indices = [index for _, index in sorted_list]
    return indices