from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from tensorboardX import x2num


class NumpyTest(unittest.TestCase):
    def test_scalar(self):
        res = x2num.make_np(1.1)
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(1 << 64 - 1)  # uint64_max
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.float16(1.00000087))
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.float128(1.00008 + 9))
        assert isinstance(res, np.ndarray) and res.shape == (1,)
        res = x2num.make_np(np.int64(100000000000))
        assert isinstance(res, np.ndarray) and res.shape == (1,)

    def test_make_grid(self):
        pass

    def test_numpy_vid(self):
        shapes = [(16, 3, 30, 28, 28), (19, 3, 30, 28, 28), (19, 3, 29, 23, 19)]
        for s in shapes:
            x = np.random.random_sample(s)
            # assert x2num.make_np(x, 'VID').shape[3] == 3

    def test_numpy_vid_uint8(self):
        x = np.random.randint(0, 256, (16, 3, 30, 28, 28)).astype(np.uint8)
        # x2num.make_np(x, 'VID').shape[3] == 3
