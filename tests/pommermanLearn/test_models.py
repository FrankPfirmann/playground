import pytest
import numpy as np
from models import BoardTracker

global_view_step_0 = np.zeros((5,5))
local_view         = np.ones((3,3))
position           = [1, 2]
forgetfullness     = 0.5
positions = [
    [1, 2],
    [1, 3],
    [2, 3]
]
local_views = np.array([
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ],
    [
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ],
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ]
], np.float64)
position_fog   = [0, 1]
local_view_fog = np.array([
    [0, 0, 0,],
    [0, 0, 1,],
    [0, 1, 0,]
], np.float64)

position_edge = [2,0]
view_edge = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
], np.float64)
position_center = [2,2]
view_center = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
], np.float64)

global_views = np.array([
    [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 1, 0, 1],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 2, 0, 2],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ]
], dtype=np.float64)
_global_views = np.array([
    [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 1, 0, 1],
        [0, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0.5, 0, 0.5],
        [0, 0.25, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ]
], dtype=np.float64)


class TestBoardTracker:
    def test_update_steps(self):
        tracker = BoardTracker((5, 5))

        tracker.update(local_views[0], positions[0])
        actual = tracker.get_view()
        assert np.array_equal(actual, global_views[0])

        tracker.update(local_views[1], positions[1])
        actual = tracker.get_view()
        assert np.array_equal(actual, global_views[1])

        tracker.update(local_views[2], positions[2])
        actual = tracker.get_view()
        assert np.array_equal(actual, global_views[2])
    
    def test_get_view_full(self):
        tracker = BoardTracker((5, 5))
        tracker.update(local_views[0], positions[0])
        actual = tracker.get_view(position=None, view_range=0, centralized=True)
        print(actual)

        assert np.array_equal(actual, view_center)

    def test_get_view_full_edge(self):
        tracker = BoardTracker((5, 5))
        tracker.update(local_views[0], positions[0])
        actual = tracker.get_view(position=position_edge, view_range=0, centralized=True)

        assert np.array_equal(actual, view_edge)
    
    def test_get_view_inside_fov(self):
        tracker = BoardTracker((5, 5))
        tracker.update(local_views[0], positions[0])
        actual = tracker.get_view(positions[0], 1, centralized=True)

        assert np.array_equal(actual, local_views[0])

    def test_get_view_outside_fov(self):
        tracker = BoardTracker((5, 5))
        tracker.update(local_views[0], positions[0])
        actual = tracker.get_view(position_fog, 1, centralized=True)

        assert np.array_equal(actual, local_view_fog)

    def test_reset_resets(self):
        tracker = BoardTracker((5, 5))
        tracker.update(local_views[0], positions[0])
        tracker.reset()

        actual = tracker.get_view()
        expected = np.zeros((5,5))

        assert np.array_equal(actual, expected)