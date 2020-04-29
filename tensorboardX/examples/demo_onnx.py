from tensorboardX import SummaryWriter

import subprocess
zoo_address = 'https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz'

res = subprocess.call(['wget', '-nc', zoo_address])
assert res == 0, 'cannot download example onnx model from the zoo'
res = subprocess.call(['tar', 'xf', 'mnist.tar.gz', '-C', 'examples/', 'mnist/model.onnx'])



with SummaryWriter() as w:
    w.add_onnx_graph('examples/mnist/model.onnx')
    # w.add_onnx_graph('/Users/dexter/Downloads/resnet50/model.onnx')
