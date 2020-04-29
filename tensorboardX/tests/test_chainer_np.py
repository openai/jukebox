from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorboardX import x2num, SummaryWriter
try:
    import chainer
    chainer_installed = True
except ImportError:
    print('Chainer is not installed, skipping test')
    chainer_installed = False
import numpy as np
import unittest


if chainer_installed:
    chainer.Variable
    tensors = [chainer.Variable(np.random.rand(3, 10, 10)),
               chainer.Variable(np.random.rand(1)),
               chainer.Variable(np.random.rand(1, 2, 3, 4, 5))]

    class ChainerTest(unittest.TestCase):
        def test_chainer_np(self):
            for tensor in tensors:
                # regular variable
                assert isinstance(x2num.make_np(tensor), np.ndarray)

            # python primitive type
            assert(isinstance(x2num.make_np(0), np.ndarray))
            assert(isinstance(x2num.make_np(0.1), np.ndarray))

        def test_chainer_img(self):
            shapes = [(77, 3, 13, 7), (77, 1, 13, 7), (3, 13, 7), (1, 13, 7), (13, 7)]
            for s in shapes:
                x = chainer.Variable(np.random.random_sample(s))
                # assert x2num.make_np(x, 'IMG').shape[2] == 3

        def test_chainer_write(self):
            with SummaryWriter() as w:
                w.add_scalar('scalar', chainer.Variable(np.random.rand(1)), 0)
