import unittest
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tensorboardX import summary
from .expect_reader import compare_proto

np.random.seed(0)
true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]


class PRCurveTest(unittest.TestCase):
    def test_smoke(self):
        with SummaryWriter() as writer:
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(
                100), 1)
            writer.add_pr_curve_raw('prcurve with raw data',
                                    true_positive_counts,
                                    false_positive_counts,
                                    true_negative_counts,
                                    false_negative_counts,
                                    precision,
                                    recall,
                                    1)

    def test_pr_purve(self):
        random_labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1,
            1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0,
            1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])
        random_probs = np.array([0.33327776, 0.30032885, 0.79012837, 0.04306813, 0.65221544,
            0.58481968, 0.28305522, 0.53795795, 0.00729739, 0.52266951,
            0.22464247, 0.11262435, 0.41573075, 0.92493992, 0.73066758,
            0.43867735, 0.27955449, 0.56975382, 0.53933028, 0.34392824,
            0.30312509, 0.81732807, 0.55408544, 0.3969487 , 0.31768033,
            0.24353266, 0.47198005, 0.19999122, 0.05788022, 0.24046305,
            0.04651082, 0.30061738, 0.78321545, 0.82670207, 0.49200517,
            0.80904619, 0.96711993, 0.3160946 , 0.01049424, 0.60108337,
            0.56508792, 0.83729429, 0.9717386 , 0.46306053, 0.80232138,
            0.24166823, 0.7393237 , 0.50820418, 0.04944932, 0.53854157,
            0.10765172, 0.84723855, 0.20518299, 0.3143431 , 0.51299074,
            0.47065695, 0.54267833, 0.1812676 , 0.06265177, 0.34110327,
            0.30915171, 0.91870169, 0.91309447, 0.31395817, 0.36780571,
            0.98297986, 0.00594547, 0.52839042, 0.70229202, 0.37779588,
            0.15207045, 0.59759632, 0.72397032, 0.71502195, 0.90135725,
            0.43970107, 0.17123532, 0.08785938, 0.04986818, 0.62702444,
            0.69171023, 0.30537792, 0.30285433, 0.27124347, 0.27693729,
            0.7136039 , 0.48022489, 0.20916285, 0.2018599 , 0.92401008,
            0.30189681, 0.46862626, 0.96353024, 0.30468533, 0.68281294,
            0.30623562, 0.40795975, 0.76824531, 0.89824215, 0.69845035], dtype=np.float16)
        compare_proto(summary.pr_curve('tag', random_labels, random_probs, 1), self)

    def test_pr_purve_raw(self):
        compare_proto(summary.pr_curve_raw('prcurve with raw data',
                                           true_positive_counts,
                                           false_positive_counts,
                                           true_negative_counts,
                                           false_negative_counts,
                                           precision,
                                           recall,
                                           1),
                      self)
