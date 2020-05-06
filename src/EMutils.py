# -*- coding: utf-8 -*-

import numpy as np

def relative_distance(new_vector, old_vector, epsilon=1e-10, use_norm=None):
    norm_old_vector = np.linalg.norm(old_vector, use_norm)
    if norm_old_vector < epsilon :
        norm_old_vector = epsilon
    return np.linalg.norm(new_vector - old_vector, use_norm) / norm_old_vector