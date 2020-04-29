import unittest
import torch
from tensorboardX import SummaryWriter


class EmbeddingTest(unittest.TestCase):
    def test_embedding(self):
        w = SummaryWriter()
        all_features = torch.Tensor([[1, 2, 3], [5, 4, 1], [3, 7, 7]])
        all_labels = torch.Tensor([33, 44, 55])
        all_images = torch.zeros(3, 3, 5, 5)

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)

        dataset_label = ['test'] * 2 + ['train'] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)
        # assert...

    def test_embedding_64(self):
        w = SummaryWriter()
        all_features = torch.Tensor([[1, 2, 3], [5, 4, 1], [3, 7, 7]])
        all_labels = torch.Tensor([33, 44, 55])
        all_images = torch.zeros((3, 3, 5, 5), dtype=torch.float64)

        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        global_step=2)

        dataset_label = ['test'] * 2 + ['train'] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(all_features,
                        metadata=all_labels,
                        label_img=all_images,
                        metadata_header=['digit', 'dataset'],
                        global_step=2)

    def test_embedding_square(self):
        w = SummaryWriter(comment='sq')
        all_features = torch.rand(228,256)
        all_images = torch.rand(228, 3, 32, 32)
        for i in range(all_images.shape[0]):
            all_images[i] *= (float(i)+60)/(all_images.shape[0]+60)
        w.add_embedding(all_features,
                        label_img=all_images,
                        global_step=2)

    def test_embedding_fail(self):
        with self.assertRaises(AssertionError):
            w = SummaryWriter(comment='shouldfail')
            all_features = torch.rand(228,256)
            all_images = torch.rand(228, 3, 16, 32)
            for i in range(all_images.shape[0]):
                all_images[i] *= (float(i)+60)/(all_images.shape[0]+60)
            w.add_embedding(all_features,
                            label_img=all_images,
                            global_step=2)
