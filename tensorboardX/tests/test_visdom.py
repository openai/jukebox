from tensorboardX import TorchVis

import numpy as np
import pytest
import unittest

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]


class VisdomTest(unittest.TestCase):
    def test_TorchVis(self):
        w = TorchVis('visdom')
        w.add_scalar('scalar_visdom', 1, 0)
        w.add_scalar('scalar_visdom', 2, 1)
        w.add_histogram('histogram_visdom', np.array([1, 2, 3, 4, 5]), 1)
        w.add_image('image_visdom', np.ndarray((3, 20, 20)), 2)
        # w.add_video('video_visdom', np.ndarray((1, 3, 10, 20, 20)), 3)
        w.add_audio('audio_visdom', [1, 2, 3, 4, 5])
        w.add_text('text_visdom', 'mystring')
        w.add_pr_curve('pr_curve_visdom', np.random.randint(2, size=100), np.random.rand(100), 10)
        w.add_pr_curve_raw('prcurve with raw data',
                           true_positive_counts,
                           false_positive_counts,
                           true_negative_counts,
                           false_negative_counts,
                           precision,
                           recall, 20)
        del w
