import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import datetime

resnet18 = models.resnet18(False)
writer = SummaryWriter()
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]


for n_iter in range(100):
    s1 = torch.rand(1)  # value to keep
    s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar('data/scalar_systemtime', s1[0], n_iter)
    # data grouping by `slash`
    writer.add_scalar('data/scalar_customtime', s1[0], n_iter, walltime=n_iter)
    writer.add_scalars('data/scalar_group', {"xsinx": n_iter * np.sin(n_iter),
                                             "xcosx": n_iter * np.cos(n_iter),
                                             "arctanx": np.arctan(n_iter)}, n_iter)
    x = torch.rand(32, 3, 64, 64)  # output from network
    if n_iter % 10 == 0:
        x = vutils.make_grid(x, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)  # Tensor
        writer.add_image_with_boxes('imagebox_label', torch.ones(3, 240, 240) * 0.5,
             torch.Tensor([[10, 10, 100, 100], [101, 101, 200, 200]]),
             n_iter, 
             labels=['abcde' + str(n_iter), 'fgh' + str(n_iter)])
        x = torch.zeros(sample_rate * 2)
        for i in range(x.size(0)):
            # sound amplitude should in [-1, 1]
            x[i] = np.cos(freqs[n_iter // 10] * np.pi *
                          float(i) / float(sample_rate))
        writer.add_audio('myAudio', x, n_iter)
        writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
        writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)
        for name, param in resnet18.named_parameters():
            if 'bn' not in name:
                writer.add_histogram(name, param, n_iter)
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

dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
writer.add_embedding(features, global_step=1, tag='noMetadata')
dataset = datasets.MNIST('mnist', train=True, download=True)
images_train = dataset.train_data[:100].float()
labels_train = dataset.train_labels[:100]
features_train = images_train.view(100, 784)

all_features = torch.cat((features, features_train))
all_labels = torch.cat((label, labels_train))
all_images = torch.cat((images, images_train))
dataset_label = ['test'] * 100 + ['train'] * 100
all_labels = list(zip(all_labels, dataset_label))

writer.add_embedding(all_features, metadata=all_labels, label_img=all_images.unsqueeze(1),
                     metadata_header=['digit', 'dataset'], global_step=2)

# VIDEO
vid_images = dataset.train_data[:16 * 48]
vid = vid_images.view(16, 48, 1, 28, 28)  # BxTxCxHxW

writer.add_video('video', vid_tensor=vid)
writer.add_video('video_1_fps', vid_tensor=vid, fps=1)

writer.close()
