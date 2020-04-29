# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple MNIST classifier to demonstrate features of Beholder.

Based on tensorflow/examples/tutorials/mnist/mnist_with_summaries.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorboardX.beholder as beholder_lib
import time

from collections import namedtuple


LOG_DIRECTORY = '/tmp/beholder-demo'
tensor_and_name = namedtuple('tensor_and_name', 'tensor, name')


def beholder_pytorch():
    for i in range(1000):
        fake_param = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i))
                      for i in range(5)]
        arrays = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i))
                  for i in range(5)]
        beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)
        beholder.update(
            trainable=fake_param,
            arrays=arrays,
            frame=np.random.randn(128, 128),
        )
        time.sleep(0.1)
        print(i)


if __name__ == '__main__':
    import os
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)
    print(LOG_DIRECTORY)
    beholder_pytorch()
