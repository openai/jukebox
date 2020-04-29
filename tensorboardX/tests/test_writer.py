from tensorboardX import SummaryWriter
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import PyRecordReader_New
from tensorboardX.proto import event_pb2

import numpy as np
import pytest
import unittest
import time
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]


class WriterTest(unittest.TestCase):
    def test_flush(self):
        N_TEST = 5
        w = SummaryWriter(flush_secs=1)
        f = w.file_writer.event_writer._ev_writer._file_name
        for i in range(N_TEST):
            w.add_scalar('a', i)
            time.sleep(2)
        r = PyRecordReader_New(f)
        r.GetNext()  # meta data, so skip
        for _ in range(N_TEST):  # all of the data should be flushed
            r.GetNext()

    def test_flush_timer_is_long_so_data_is_not_there(self):
        with self.assertRaises(BaseException):
            N_TEST = 5
            w = SummaryWriter(flush_secs=20)
            f = w.file_writer.event_writer._ev_writer._file_name
            for i in range(N_TEST):
                w.add_scalar('a', i)
                time.sleep(2)
            r = PyRecordReader_New(f)
            r.GetNext()  # meta data, so skip
            for _ in range(N_TEST):  # missing data
                r.GetNext()

    def test_flush_after_close(self):
        N_TEST = 5
        w = SummaryWriter(flush_secs=20)
        f = w.file_writer.event_writer._ev_writer._file_name
        for i in range(N_TEST):
            w.add_scalar('a', i)
            time.sleep(2)
        w.close()
        r = PyRecordReader_New(f)
        r.GetNext()  # meta data, so skip
        for _ in range(N_TEST):  # all of the data should be flushed
            r.GetNext()

    def test_flush(self):
        N_TEST = 5
        w = SummaryWriter(flush_secs=20)
        f = w.file_writer.event_writer._ev_writer._file_name
        for i in range(N_TEST):
            w.add_scalar('a', i)
            time.sleep(2)
        w.flush()
        r = PyRecordReader_New(f)
        r.GetNext()  # meta data, so skip
        for _ in range(N_TEST):  # all of the data should be flushed
            r.GetNext()

    def test_auto_close(self):
        pass

    def test_writer(self):
        with SummaryWriter() as writer:
            sample_rate = 44100

            n_iter = 0
            writer.add_scalar('data/scalar_systemtime', 0.1, n_iter)
            writer.add_scalar('data/scalar_customtime', 0.2, n_iter, walltime=n_iter)
            writer.add_scalars('data/scalar_group', {"xsinx": n_iter * np.sin(n_iter),
                                                     "xcosx": n_iter * np.cos(n_iter),
                                                     "arctanx": np.arctan(n_iter)}, n_iter)
            x = np.zeros((32, 3, 64, 64))  # output from network
            writer.add_images('Image', x, n_iter)  # Tensor
            writer.add_image_with_boxes('imagebox',
                                        np.zeros((3, 64, 64)),
                                        np.array([[10, 10, 40, 40], [40, 40, 60, 60]]),
                                        n_iter)
            x = np.zeros(sample_rate * 2)

            writer.add_audio('myAudio', x, n_iter)
            writer.add_video('myVideo', np.random.rand(16, 48, 1, 28, 28).astype(np.float32), n_iter)
            writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
            writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)
            writer.add_histogram('hist', np.random.rand(100, 100), n_iter)
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(
                100), n_iter)  # needs tensorboard 0.4RC or later
            writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
                                    false_positive_counts,
                                    true_negative_counts,
                                    false_negative_counts,
                                    precision,
                                    recall, n_iter)
            # export scalar data to JSON for external processing
            writer.export_scalars_to_json("./all_scalars.json")
            imgs = []
            for i in range(5):
                imgs.append(np.ones((3, 100, 110)))
            with SummaryWriter() as w:
                w.add_images('img_list', imgs, dataformats='CHW')