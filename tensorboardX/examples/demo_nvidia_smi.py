"""
write gpu and (gpu) memory usage of nvidia cards as scalar
"""
from tensorboardX import SummaryWriter
import time
import torch
try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # gpu0
except ImportError:
    print('This demo needs nvidia-ml-py or nvidia-ml-py3')
    exit()


with SummaryWriter() as writer:
    x = []
    for n_iter in range(50):
        x.append(torch.Tensor(1000, 1000).cuda())
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        writer.add_scalar('nv/gpu', res.gpu, n_iter)
        res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        writer.add_scalar('nv/gpu_mem', res.used, n_iter)
        time.sleep(0.1)
