import torch as t
import jukebox.utils.dist_adapter as dist
from tqdm import tqdm
from datetime import date
import os
import sys

def def_tqdm(x):
    return tqdm(x, leave=True, file=sys.stdout, bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

def get_range(x):
    if dist.get_rank() == 0:
        return def_tqdm(x)
    else:
        return x

def init_logging(hps, local_rank, rank):
    logdir = f"{hps.local_logdir}/{hps.name}"
    if local_rank == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        with open(logdir + 'argv.txt', 'w') as f:
            f.write(hps.argv + '\n')
        print("Logging to", logdir)
    logger = Logger(logdir, rank)
    metrics = Metrics()
    logger.add_text('hps', str(hps))
    return logger, metrics

def get_name(hps):
    name = ""
    for key, value in hps.items():
        name += f"{key}_{value}_"
    return name

def average_metrics(_metrics):
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
    return {key: sum(vals)/len(vals) for key, vals in metrics.items()}

class Metrics:
    def __init__(self):
        self.sum = {}
        self.n = {}

    def update(self, tag, val, batch):
        # v is average value over batch
        # store total value and total batch, returns dist average
        sum = t.tensor(val * batch).float().cuda()
        n = t.tensor(batch).float().cuda()
        dist.all_reduce(sum)
        dist.all_reduce(n)
        sum = sum.item()
        n = n.item()
        self.sum[tag] = self.sum.get(tag, 0.0) + sum
        self.n[tag] = self.n.get(tag, 0.0) + n
        return sum / n

    def avg(self, tag):
        if tag in self.sum:
            return self.sum[tag] / self.n[tag]
        else:
            return 0.0

    def reset(self):
        self.sum = {}
        self.n = {}

class Logger:
    def __init__(self, logdir, rank):
        if rank == 0:
            from tensorboardX import SummaryWriter
            self.sw = SummaryWriter(f"{logdir}/logs")
        self.iters = 0
        self.rank = rank
        self.works = []
        self.logdir = logdir

    def step(self):
        self.iters += 1

    def flush(self):
        if self.rank == 0:
            self.sw.flush()

    def add_text(self, tag, text):
        if self.rank == 0:
            self.sw.add_text(tag, text, self.iters)

    def add_audios(self, tag, auds, sample_rate=22050, max_len=None, max_log=8):
        if self.rank == 0:
            for i in range(min(len(auds), max_log)):
                if max_len:
                    self.sw.add_audio(f"{i}/{tag}", auds[i][:max_len * sample_rate], self.iters, sample_rate)
                else:
                    self.sw.add_audio(f"{i}/{tag}", auds[i], self.iters, sample_rate)

    def add_audio(self, tag, aud, sample_rate=22050):
        if self.rank == 0:
            self.sw.add_audio(tag, aud, self.iters, sample_rate)

    def add_images(self, tag, img, dataformats="NHWC"):
        if self.rank == 0:
            self.sw.add_images(tag, img, self.iters, dataformats=dataformats)

    def add_image(self, tag, img):
        if self.rank == 0:
            self.sw.add_image(tag, img, self.iters)

    def add_scalar(self, tag, val):
        if self.rank == 0:
            self.sw.add_scalar(tag, val, self.iters)

    def get_range(self, loader):
        if self.rank == 0:
            self.trange = def_tqdm(loader)
        else:
            self.trange = loader
        return enumerate(self.trange)

    def close_range(self):
        if self.rank == 0:
            self.trange.close()

    def set_postfix(self, *args, **kwargs):
        if self.rank == 0:
            self.trange.set_postfix(*args, **kwargs)

    # For logging summaries of varies graph ops
    def add_reduce_scalar(self, tag, layer, val):
        if self.iters % 100 == 0:
            with t.no_grad():
                val = val.float().norm()/float(val.numel())
            work = dist.reduce(val, 0, async_op=True)
            self.works.append((tag, layer, val, work))

    def finish_reduce(self):
        for tag, layer, val, work in self.works:
            work.wait()
            if self.rank == 0:
                val = val.item()/dist.get_world_size()
                self.lw[layer].add_scalar(tag, val, self.iters)
        self.works = []
