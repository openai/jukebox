from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorboardX import x2num, SummaryWriter
import torch
import numpy as np
import unittest


class PyTorchNumpyTest(unittest.TestCase):
    def test_pytorch_np(self):
        tensors = [torch.rand(3, 10, 10), torch.rand(1), torch.rand(1, 2, 3, 4, 5)]
        for tensor in tensors:
            # regular tensor
            assert isinstance(x2num.make_np(tensor), np.ndarray)

            # CUDA tensor
            if torch.cuda.device_count() > 0:
                assert isinstance(x2num.make_np(tensor.cuda()), np.ndarray)

            # regular variable
            assert isinstance(x2num.make_np(torch.autograd.Variable(tensor)), np.ndarray)

            # CUDA variable
            if torch.cuda.device_count() > 0:
                assert isinstance(x2num.make_np(torch.autograd.Variable(tensor).cuda()), np.ndarray)

        # python primitive type
        assert(isinstance(x2num.make_np(0), np.ndarray))
        assert(isinstance(x2num.make_np(0.1), np.ndarray))

    def test_pytorch_write(self):
        with SummaryWriter() as w:
            w.add_scalar('scalar', torch.autograd.Variable(torch.rand(1)), 0)

    def test_pytorch_histogram(self):
        with SummaryWriter() as w:
            w.add_histogram('float histogram', torch.rand((50,)))
            w.add_histogram('int histogram', torch.randint(0, 100, (50,)))

    def test_pytorch_histogram_raw(self):
        with SummaryWriter() as w:
            num = 50
            floats = x2num.make_np(torch.rand((num,)))
            bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            counts, limits = np.histogram(floats, bins)
            sum_sq = floats.dot(floats).item()
            w.add_histogram_raw('float histogram raw',
                                min=floats.min().item(),
                                max=floats.max().item(),
                                num=num,
                                sum=floats.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits[1:].tolist(),
                                bucket_counts=counts.tolist())

            ints = x2num.make_np(torch.randint(0, 100, (num,)))
            bins = [0, 25, 50, 75, 100]
            counts, limits = np.histogram(ints, bins)
            sum_sq = ints.dot(ints).item()
            w.add_histogram_raw('int histogram raw',
                                min=ints.min().item(),
                                max=ints.max().item(),
                                num=num,
                                sum=ints.sum().item(),
                                sum_squares=sum_sq,
                                bucket_limits=limits[1:].tolist(),
                                bucket_counts=counts.tolist())
