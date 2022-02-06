import pytest
import os
import pickle
import numpy as np
from pommermanLearn.util.data import calculate_center, crop_view, decentralize_view, merge_views, centralize_view, transform_observation

RES_DIR = os.path.dirname(os.path.abspath(__file__))

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
expected_merge=np.array([
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
    assert np.array_equal(actual, expected_merge)

def test_merge_planes_forgetfullness_one():
    merge_views(first, second, fov, 1.0)

def test_merge_planes_forgetfullness_negative():
    with pytest.raises(Exception):
        merge_views(first, second, fov, -0.1)

def test_merge_planes_forgetfullness_over_one():
    with pytest.raises(Exception):
        merge_views(first, second, fov, 1.1)
    
# centralize_plane
position = [1, 2]
plane_decentralized = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
plane_centralized_padding_zeros = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 2, 0, 0],
    [3, 4, 5, 0, 0],
    [6, 7, 8, 0, 0],
    [0, 0, 0, 0, 0]
])
plane_centralized_padding_ones = np.array([
    [1, 1, 1, 1, 1],
    [0, 1, 2, 1, 1],
    [3, 4, 5, 1, 1],
    [6, 7, 8, 1, 1],
    [1, 1, 1, 1, 1]
])

def test_centralize_plane_padding_zeros():
    actual = centralize_view(plane_decentralized, position=position, padding=0)
    print(actual)
    print(plane_centralized_padding_zeros)
    assert np.array_equal(actual, plane_centralized_padding_zeros)

def test_centralize_plane_padding_ones():
    actual = centralize_view(plane_decentralized, position=position, padding=1)
    assert np.array_equal(actual, plane_centralized_padding_ones)

def test_decentralize_view():
    actual = decentralize_view(plane_centralized_padding_zeros, position, plane_decentralized.shape)
    assert np.array_equal(actual, plane_decentralized)

# test_crop_view
view_full       = np.arange(25).reshape(5,5)
view_range_good = 1
view_range_bad  = 3
view_cropped    = np.array([
    [6,  7,  8],
    [11, 12, 13],
    [16, 17, 18]
])

def test_crop_view_reject_even_width():
    with pytest.raises(ValueError):
        crop_view(np.ones((4,5)), view_range_good)
    
def test_crop_view_reject_even_height():
    with pytest.raises(ValueError):
        crop_view(np.ones((5,4)), view_range_good)

def test_crop_view_reject_even_width_and_height():
    with pytest.raises(ValueError):
        crop_view(np.ones((4,4)), view_range_good)

def test_crop_view_reject_view_range_negative():
    with pytest.raises(ValueError):
        crop_view(view_full, -1)

def test_crop_view_reject_view_range_zero():
    with pytest.raises(ValueError):
        crop_view(view_full, 0)

def test_crop_view_reject_view_range_out_of_bounds():
    with pytest.raises(ValueError):
        crop_view(view_full, view_range_bad)

def test_crop_view():
    actual = crop_view(view_full, view_range_good)
    assert np.array_equal(actual, view_cropped)

# transform_observation
obs         = pickle.load(open(os.path.join(RES_DIR, "obs.pkl"), 'rb'))
expected_ff = np.load(os.path.join(RES_DIR, "expected_ff.npy"), allow_pickle=True)
expected_ft = np.load(os.path.join(RES_DIR, "expected_ft.npy"), allow_pickle=True)
expected_tt = np.load(os.path.join(RES_DIR, "expected_tt.npy"), allow_pickle=True)

def test_transform_observation_ff():
    actual = transform_observation(obs, p_obs=False, centralized=False)
    assert np.array_equal(actual, expected_ff)

def test_transform_observation_ft():
    actual = transform_observation(obs, p_obs=False, centralized=True)
    assert np.array_equal(actual, expected_ft)

def test_transform_observation_tf():
    with pytest.raises(ValueError):
        transform_observation(obs, p_obs=True, centralized=False)

def test_transform_observation_tt():
    actual = transform_observation(obs, p_obs=True, centralized=True)
    assert np.array_equal(actual, expected_tt)

# calculate_center
def test_calculate_center_rejects_even_width():
    with pytest.raises(ValueError):
        calculate_center((5, 4))

def test_calculate_center_rejects_even_height():
    with pytest.raises(ValueError):
        calculate_center((4, 5))

def test_calculate_center_rejects_even_width_and_height():
    with pytest.raises(ValueError):
        calculate_center((4, 4))

def test_calculate_center_x():
    x, _ = calculate_center((5, 7))
    assert x == 2

def test_calculate_center_y():
    _, y = calculate_center((5, 7))
    assert y == 3
