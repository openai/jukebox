from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import unittest

from tensorboardX import SummaryWriter


class FigureTest(unittest.TestCase):
    def test_figure(self):
        writer = SummaryWriter()

        figure, axes = plt.figure(), plt.gca()
        circle1 = plt.Circle((0.2, 0.5), 0.2, color='r')
        circle2 = plt.Circle((0.8, 0.5), 0.2, color='g')
        axes.add_patch(circle1)
        axes.add_patch(circle2)
        plt.axis('scaled')
        plt.tight_layout()

        writer.add_figure("add_figure/figure", figure, 0, close=False)
        assert plt.fignum_exists(figure.number) is True

        writer.add_figure("add_figure/figure", figure, 1)
        assert plt.fignum_exists(figure.number) is False

        writer.close()

    def test_figure_list(self):
        writer = SummaryWriter()

        figures = []
        for i in range(5):
            figure = plt.figure()
            plt.plot([i * 1, i * 2, i * 3], label="Plot " + str(i))
            plt.xlabel("X")
            plt.xlabel("Y")
            plt.legend()
            plt.tight_layout()
            figures.append(figure)

        writer.add_figure("add_figure/figure_list", figures, 0, close=False)
        assert all([plt.fignum_exists(figure.number) is True for figure in figures])

        writer.add_figure("add_figure/figure_list", figures, 1)
        assert all([plt.fignum_exists(figure.number) is False for figure in figures])

        writer.close()
