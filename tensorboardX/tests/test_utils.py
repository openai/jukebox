from tensorboardX import summary
from tensorboardX.utils import make_grid, _prepare_video, convert_to_HWC
import numpy as np
import pytest
import unittest


class UtilsTest(unittest.TestCase):
    def test_to_HWC(self):
        np.random.seed(1)
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'chw')
        assert converted.shape == (32, 32, 3)
        test_image = np.random.randint(0, 256, size=(16, 3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'nchw')
        assert converted.shape == (64, 256, 3)
        test_image = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, 'hw')
        assert converted.shape == (32, 32, 3)

    def test_prepare_video(self):
        # at each timestep the sum over all other dimensions of the video should stay the same
        np.random.seed(1)
        V_before = np.random.random((4, 10, 3, 20, 20))
        V_after = _prepare_video(np.copy(V_before))
        V_before = np.swapaxes(V_before, 0, 1)
        V_before = np.reshape(V_before, newshape=(10, -1))
        V_after = np.reshape(V_after, newshape=(10, -1))
        np.testing.assert_array_almost_equal(np.sum(V_before, axis=1), np.sum(V_after, axis=1))
