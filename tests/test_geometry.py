"""Minimal sanity tests for geometry helpers.

Checks:
- space_vector maps a grid index to the voxel center (0.5 * scale)
- max_box_index returns the opposite corner in a cubic grid
"""

import numpy as np

from pathlib import Path
import sys

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry import space_vector, max_box_index  # noqa: E402


def test_space_vector_center():
    """space_vector returns voxel center for index (0,0,0)."""
    idx = (0, 0, 0)
    scale = np.array([1.0, 2.0, 3.0], dtype=float) / np.array([2, 2, 2])
    v = space_vector(idx, scale)
    assert v.shape == (3,)
    np.testing.assert_allclose(v, 0.5 * scale)


def test_max_box_index_corners():
    """max_box_index returns the farthest corner in a 10x10x10 grid."""
    shape = np.array([10, 10, 10], dtype=np.int32)

    far = max_box_index(np.array([0, 0, 0], dtype=np.int32), shape)
    np.testing.assert_array_equal(far, np.array([9, 9, 9], dtype=np.int32))

    far = max_box_index(np.array([9, 9, 9], dtype=np.int32), shape)
    np.testing.assert_array_equal(far, np.array([0, 0, 0], dtype=np.int32))
