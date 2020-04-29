import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# EMBEDDING VISUALIZATION FOR A TWO-CLASSES PROBLEM

# just a bunch of layers


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.cn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.cn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(in_features=128, out_features=2)

    def forward(self, i):
        i = self.cn1(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = self.cn2(i)
        i = F.relu(i)
        i = F.max_pool2d(i, 2)
        i = i.view(len(i), -1)
        i = self.fc1(i)
        i = F.log_softmax(i, dim=1)
        return i

# get some random data around value


def get_data(value, shape):
    data = torch.ones(shape) * value
    # add some noise
    data += torch.randn(shape)**2
    return data


# dataset
# cat some data with different values
data = torch.cat(
    (get_data(
        0, (100, 1, 14, 14)), get_data(
            0.5, (100, 1, 14, 14))), 0)
# labels
labels = torch.cat((torch.zeros(100), torch.ones(100)), 0)
# generator
gen = DataLoader(TensorDataset(data, labels), batch_size=25, shuffle=True)
# network
m = M()
#loss and optim
loss = nn.NLLLoss()
optimizer = torch.optim.Adam(params=m.parameters())
# settings for train and log
num_epochs = 20
embedding_log = 5
writer = SummaryWriter(comment='mnist_embedding_training')

# TRAIN
for epoch in range(num_epochs):
    for j, sample in enumerate(gen):
        n_iter = (epoch * len(gen)) + j
        # reset grad
        m.zero_grad()
        optimizer.zero_grad()
        # get batch data
        data_batch = Variable(sample[0], requires_grad=True).float()
        label_batch = Variable(sample[1], requires_grad=False).long()
        # FORWARD
        out = m(data_batch)
        loss_value = loss(out, label_batch)
        # BACKWARD
        loss_value.backward()
        optimizer.step()
        # LOGGING
        writer.add_scalar('loss', loss_value.data.item(), n_iter)

        if j % embedding_log == 0:
            print("loss_value:{}".format(loss_value.data.item()))
            # we need 3 dimension for tensor to visualize it!
            out = torch.cat((out.data, torch.ones(len(out), 1)), 1)
            writer.add_embedding(
                out,
                metadata=label_batch.data,
                label_img=data_batch.data,
                global_step=n_iter)

writer.close()

# tensorboard --logdir runs
# you should now see a dropdown list with all the timestep,
# last timestep should have a visible separation between the two classes
