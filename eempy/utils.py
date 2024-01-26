"""
Useful functions for EEM processing
"""


import numpy as np
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