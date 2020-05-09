import numpy as np
import torch as t
import jukebox.utils.dist_adapter as dist
import soundfile
import librosa
from jukebox.utils.dist_utils import print_once

class DefaultSTFTValues:
    def __init__(self, hps):
        self.sr = hps.sr
        self.n_fft = 2048
        self.hop_length = 256
        self.window_size = 6 * self.hop_length

class STFTValues:
    def __init__(self, hps, n_fft, hop_length, window_size):
        self.sr = hps.sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size

def calculate_bandwidth(dataset, hps, duration=600):
    hps = DefaultSTFTValues(hps)
    n_samples = int(dataset.sr * duration)
    l1, total, total_sq, n_seen, idx = 0.0, 0.0, 0.0, 0.0, dist.get_rank()
    spec_norm_total, spec_nelem = 0.0, 0.0
    while n_seen < n_samples:
        x = dataset[idx]
        if isinstance(x, (tuple, list)):
            x, y = x
        samples = x.astype(np.float64)
        stft = librosa.core.stft(np.mean(samples, axis=1), hps.n_fft, hop_length=hps.hop_length, win_length=hps.window_size)
        spec = np.absolute(stft)
        spec_norm_total += np.linalg.norm(spec)
        spec_nelem += 1
        n_seen += int(np.prod(samples.shape))
        l1 += np.sum(np.abs(samples))
        total += np.sum(samples)
        total_sq += np.sum(samples ** 2)
        idx += max(16, dist.get_world_size())

    if dist.is_available():
        from jukebox.utils.dist_utils import allreduce
        n_seen = allreduce(n_seen)
        total = allreduce(total)
        total_sq = allreduce(total_sq)
        l1 = allreduce(l1)
        spec_nelem = allreduce(spec_nelem)
        spec_norm_total = allreduce(spec_norm_total)

    mean = total / n_seen
    bandwidth = dict(l2 = total_sq / n_seen - mean ** 2,
                     l1 = l1 / n_seen,
                     spec = spec_norm_total / spec_nelem)
    print_once(bandwidth)
    return bandwidth

def audio_preprocess(x, hps):
    # Extra layer in case we want to experiment with different preprocessing
    # For two channel, blend randomly into mono (standard is .5 left, .5 right)

    # x: NTC
    x = x.float()
    if x.shape[-1]==2:
        if hps.aug_blend:
            mix=t.rand((x.shape[0],1), device=x.device) #np.random.rand()
        else:
            mix = 0.5
        x=(mix*x[:,:,0]+(1-mix)*x[:,:,1])
    elif x.shape[-1]==1:
        x=x[:,:,0]
    else:
        assert False, f'Expected channels {hps.channels}. Got unknown {x.shape[-1]} channels'

    # x: NT -> NTC
    x = x.unsqueeze(2)
    return x

def audio_postprocess(x, hps):
    return x

def stft(sig, hps):
    return t.stft(sig, hps.n_fft, hps.hop_length, win_length=hps.window_size, window=t.hann_window(hps.window_size, device=sig.device))

def spec(x, hps):
    return t.norm(stft(x, hps), p=2, dim=-1)

def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()

def squeeze(x):
    if len(x.shape) == 3:
        assert x.shape[-1] in [1,2]
        x = t.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f'Unknown input shape {x.shape}')
    return x

def spectral_loss(x_in, x_out, hps):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)
    return norm(spec_in - spec_out)

def multispectral_loss(x_in, x_out, hps):
    losses = []
    assert len(hps.multispec_loss_n_fft) == len(hps.multispec_loss_hop_length) == len(hps.multispec_loss_window_size)
    args = [hps.multispec_loss_n_fft,
            hps.multispec_loss_hop_length,
            hps.multispec_loss_window_size]
    for n_fft, hop_length, window_size in zip(*args):
        hps = STFTValues(hps, n_fft, hop_length, window_size)
        spec_in = spec(squeeze(x_in.float()), hps)
        spec_out = spec(squeeze(x_out.float()), hps)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)

def spectral_convergence(x_in, x_out, hps, epsilon=2e-3):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)

    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return (residual_norm * mask) / t.clamp(gt_norm, min=epsilon)

def log_magnitude_loss(x_in, x_out, hps, epsilon=1e-4):
    hps = DefaultSTFTValues(hps)
    spec_in = t.log(spec(squeeze(x_in.float()), hps) + epsilon)
    spec_out = t.log(spec(squeeze(x_out.float()), hps) + epsilon)
    return t.mean(t.abs(spec_in - spec_out))

def load_audio(file, sr, offset, duration, mono=False):
    # Librosa loads more filetypes than soundfile
    x, _ = librosa.load(file, sr=sr, mono=mono, offset=offset/sr, duration=duration/sr)
    if len(x.shape) == 1:
        x = x.reshape((1, -1))
    return x    


def save_wav(fname, aud, sr):
    # clip before saving?
    aud = t.clamp(aud, -1, 1).cpu().numpy()
    for i in list(range(aud.shape[0])):
        soundfile.write(f'{fname}/item_{i}.wav', aud[i], samplerate=sr, format='wav')


