import unittest
import torch
from tensorboardX import SummaryWriter


class ONNXGraphTest(unittest.TestCase):
    def test_onnx_graph(self):
        import subprocess
        zoo_address = 'https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz'

        res = subprocess.call(['wget', '-nc', zoo_address])
        assert res == 0, 'cannot download example onnx model from the zoo'
        res = subprocess.call(['tar', 'xf', 'mnist.tar.gz', '-C', 'examples/', 'mnist/model.onnx'])

        with SummaryWriter() as w:
            w.add_onnx_graph('examples/mnist/model.onnx')
