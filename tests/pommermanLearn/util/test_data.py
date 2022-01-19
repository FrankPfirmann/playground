import pytest
import numpy as np
from pommermanLearn.util.data import merge_views

first=np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])
second=np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])

wrong_shape=np.ones((4,3))

fov=np.array([
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

forgetfullness=0.2
expected=np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.4, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])
def test_merge_planes_matching_shapes():
    merge_views(first, second, fov)

def test_merge_planes_first_wrong_shape():
    with pytest.raises(Exception):
        merge_views(wrong_shape, second, fov)

def test_merge_planes_second_wrong_shape():
    with pytest.raises(Exception):
        merge_views(first, wrong_shape, fov)

def test_merge_planes_fov_wrong_shape():
    with pytest.raises(Exception):
        merge_views(first, second, wrong_shape)

def test_merge_planes_forgetfullness_zero():
    merge_views(first, second, fov, 0.0)

def test_merge_planes_forgetfullness_inbetween():
    actual = merge_views(first, second, fov, forgetfullness=forgetfullness)
    assert np.array_equal(actual, expected)

def test_merge_planes_forgetfullness_one():
    merge_views(first, second, fov, 1.0)

def test_merge_planes_forgetfullness_negative():
    with pytest.raises(Exception):
        merge_views(first, second, fov, -0.1)

def test_merge_planes_forgetfullness_over_one():
    with pytest.raises(Exception):
        merge_views(first, second, fov, 1.1)