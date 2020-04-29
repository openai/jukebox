import torch as t
import numpy as np
from jukebox.sample import sample_level
from jukebox.utils.torch_utils import assert_shape
from jukebox.hparams import Hyperparams

def repeat(x, n, dim):
    if dim == -1:
        dim = len(x.shape) - 1
    return x.reshape(int(np.prod(x.shape[:dim+1])), 1, int(np.prod(x.shape[dim+1:]))).repeat(1,n,1).reshape(*x.shape[:dim], n * x.shape[dim], *x.shape[dim+1:])

# Tests
class DummyPrior:
    def __init__(self, n_ctx, level, levels):
        self.n_ctx = n_ctx
        self.level = level
        self.levels = levels
        self.downsamples = (8,4,4)
        self.cond_downsample = self.downsamples[level+1] if level != self.levels - 1 else None
        self.raw_to_tokens = int(np.prod(self.downsamples[:level+1]))
        self.sample_length = self.n_ctx*self.raw_to_tokens

        print(f"Level:{level}, Cond downsample:{self.cond_downsample}, Raw to tokens:{self.raw_to_tokens}, Sample length:{self.sample_length}")

    def get_y(self, labels, start):
        y = labels['y'].clone()
        # Set sample_length to match this level
        y[:, 2] = self.sample_length
        # Set offset
        y[:, 1:2] = y[:, 1:2] + start * self.raw_to_tokens
        return y

    def get_z_conds(self, zs, start, end):
        if self.level != self.levels - 1:
            assert start % self.cond_downsample == end % self.cond_downsample == 0
            z_cond = zs[self.level + 1][:,start//self.cond_downsample:end//self.cond_downsample]
            assert z_cond.shape[1] == self.n_ctx//self.cond_downsample
            z_conds = [z_cond]
        else:
            z_conds = None
        return z_conds

    def ancestral_sample(self, n_samples, z_conds=None, y=None):
        z = t.zeros((n_samples, self.n_ctx), dtype=t.long, device='cuda') + \
            t.arange(0, self.n_ctx, dtype=t.long, device='cuda').view(1, self.n_ctx)

        if z_conds is not None:
            z_cond = z_conds[0]
            assert_shape(z_cond, (n_samples, self.n_ctx // 4))
            assert (z // 4 == repeat(z_cond, 4, 1)).all(), f'z: {z}, z_cond: {z_cond}, diff: {(z // 4) - repeat(z_cond, 4, 1)}'
        return z

    def primed_sample(self, n_samples, z, z_conds=None, y=None):
        prime = z.shape[1]
        assert_shape(z, (n_samples, prime))
        start = z[:,-1:] + 1
        z_rest = (t.arange(0, self.n_ctx - prime, dtype=t.long, device='cuda').view(1, self.n_ctx - prime) + start).view(n_samples, self.n_ctx - prime)
        z = t.cat([z, z_rest], dim=1)

        if z_conds is not None:
            z_cond = z_conds[0]
            assert_shape(z_cond, (n_samples, self.n_ctx // 4))
            assert (z // 4 == repeat(z_cond, 4, 1)).all(), f'z: {z}, z_cond: {z_cond}, diff: {(z // 4) - repeat(z_cond, 4, 1)}'
        return z

# Sample multiple levels
def _sample(zs, labels,  priors, sample_levels, hps):
    for level in reversed(sample_levels):
        prior = priors[level]
        # set correct total_length, hop_length and sampling_kwargs for level
        total_length = (hps.sample_length * hps.n_segment)//prior.raw_to_tokens
        hop_length = hps.hop_lengths[level]
        zs = sample_level(zs, labels[level], dict(), level, prior, total_length, hop_length, hps)
    return zs

# Ancestral sample
def test_ancestral_sample(labels, priors, hps):
    sample_levels = list(range(hps.levels))
    zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(hps.levels)]
    zs = _sample(zs, labels, priors, sample_levels, hps)

    # Test
    for z in zs:
        total_length = z.shape[1]
        # Check sample
        assert ((z - t.arange(0, total_length, dtype=t.long, device='cuda').view(1, total_length)) == 0).all()

    print("dummy ancestral sample passed")

def test_primed_sample(labels, priors, hps):
    sample_levels = list(range(hps.levels))

    start = t.tensor([15, 23, 11, 9], dtype=t.long, device='cuda').view(4, 1)

    zs_in = []
    zs = []
    for i in reversed(range(3)):
        n_ctx = 8192*(4**i)
        n_prime = n_ctx // 4
        z_prime = t.arange(0, n_prime, dtype=t.long, device='cuda').view(1, n_prime) % (2*(4**i))
        z_rest = t.randint(-10, -1, size=(1, n_ctx - n_prime), dtype=t.long, device='cuda')
        z_in = t.cat([z_prime, z_rest], dim=1) + (4**i)*start
        zs_in.append(z_in)
        zs.append(z_prime + (4**i)*start)

    zs = _sample(zs, labels, priors, sample_levels, hps)

    # Test
    for z, z_in in zip(zs, zs_in):
        total_length = z.shape[1]
        prime_length = z.shape[1] // (4 * hps.n_segment)
        # Match prime tokens
        assert (z[:,:prime_length] == z_in[:,:prime_length]).all()
        # Check sample
        z_rest = z[:,prime_length-1:] - z[:,prime_length-1:prime_length]
        assert ((z_rest - t.arange(0, total_length - prime_length + 1, dtype=t.long, device='cuda').view(1, total_length - prime_length + 1)) == 0).all()

    print("dummy primed sample passed")

def check_sample():
    n_ctx = 8192
    n_samples = 4
    levels = 3
    priors = [DummyPrior(n_ctx, level, levels) for level in range(levels)]
    max_total_length, offset, sample_length = 4134368, 0, n_ctx*8*4*4
    y = t.tensor([max_total_length, offset, sample_length, 10, 1, -1, -1, -1, -1], dtype=t.long, device='cuda').view(1, 9).repeat(n_samples, 1)
    labels = [dict(y=y, info=[[]*n_samples]) for level in range(levels)]
    hps = Hyperparams({
        'levels': 3,
        'sample_length': sample_length,
        'n_segment': 2,
        'n_ctx': n_ctx,
        'n_tokens': 0,
        'hop_lengths': [n_ctx//2, n_ctx//2, n_ctx//8],
        'n_samples': n_samples,
        'use_tokens': False
    })
    test_ancestral_sample(labels, priors, hps)
    test_primed_sample(labels, priors, hps)

check_sample()
