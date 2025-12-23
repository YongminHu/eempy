"""
Useful functions for EEM processing
"""


import numpy as np
import tensorly as tl
from tensorly.base import unfold
from scipy.spatial.distance import euclidean


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


def str_string_to_list(input_string):
    try:
        # Split the input string by commas and remove any leading/trailing spaces
        str_list = [item.strip() for item in input_string.split(",")]
        return str_list
    except ValueError:
        return None


def num_string_to_list(input_string):
    try:
        # Split the input string by commas and convert each part to an integer
        num_list = [int(num.strip()) for num in input_string.split(',')]
        return num_list
    except ValueError:
        # Handle invalid input (e.g., non-numeric characters)
        return None


def random_split_columns(arr, splits_dict, random_state=42):
    """
    Splits specified columns of a NumPy array into n_splits random components,
    preserving the original column order. Each set of split values replaces the
    original column and sums row-wise to the original value.

    Parameters:
    arr (np.ndarray): 2D array of shape (n_samples, n_features).
    columns (list): List of column indices to split.
    n_splits (int): Number of splits per column.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    np.ndarray: Modified array with split columns in original positions.
    """
    rng = np.random.default_rng(random_state)
    arr = np.asarray(arr)
    n_rows, n_cols = arr.shape
    splits_dict = dict(sorted(splits_dict.items()))
    result_cols = []
    current_col = 0
    for col, n in splits_dict.items():
        if n <= 1:
            n = 1
        # Append untouched columns before this split column
        while current_col < col:
            result_cols.append(arr[:, current_col][:, np.newaxis])
            current_col += 1

        # Split the column
        col_vals = arr[:, col]
        rand_fracs = rng.random((n_rows, n))
        rand_fracs /= rand_fracs.sum(axis=1, keepdims=True)
        split_vals = rand_fracs * col_vals[:, np.newaxis]

        # Append split columns
        for i in range(n):
            result_cols.append(split_vals[:, i][:, np.newaxis])

        current_col += 1  # Skip the original column

    # Append remaining untouched columns
    while current_col < n_cols:
        result_cols.append(arr[:, current_col][:, np.newaxis])
        current_col += 1

    return np.hstack(result_cols)


def eem_stack_to_2d(eem_stack):
    return eem_stack.reshape([eem_stack.shape[0], -1])