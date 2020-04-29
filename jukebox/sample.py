import os
import torch as t

from jukebox.hparams import Hyperparams
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav
from jukebox.make_models import make_model
from jukebox.align import get_alignment
from jukebox.save_html import save_html
import fire

def split_batch(obj, n_samples, split_size):
    n_passes = (n_samples + split_size - 1) // split_size
    if isinstance(obj, t.Tensor):
        return t.split(obj, split_size, dim=0)
    elif isinstance(obj, list):
        return list(zip(*[t.split(item, split_size, dim=0) for item in obj]))
    elif obj is None:
        return [None] * n_passes
    else:
        raise TypeError('Unknown input type')

# Sample a partial window of length<n_ctx with tokens_to_sample new tokens on level=level
def sample_partial_window(zs, labels, sampling_kwargs, level, prior, tokens_to_sample, hps):
    z = zs[level]
    n_ctx = prior.n_ctx
    current_tokens = z.shape[1]
    if current_tokens < n_ctx - tokens_to_sample:
        sampling_kwargs['sample_tokens'] = current_tokens + tokens_to_sample
        start = 0
    else:
        sampling_kwargs['sample_tokens'] = n_ctx
        start = current_tokens - n_ctx + tokens_to_sample

    return sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)

# Sample a single window of length=n_ctx at position=start on level=level
def sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps):
    n_samples = hps.n_samples
    n_ctx = prior.n_ctx
    end = start + n_ctx

    # get z already sampled at current level
    z = zs[level][:,start:end]

    if 'sample_tokens' in sampling_kwargs:
        # Support sampling a window shorter than n_ctx
        sample_tokens = sampling_kwargs['sample_tokens']
    else:
        sample_tokens = (end - start)
    conditioning_tokens, new_tokens = z.shape[1], sample_tokens - z.shape[1]

    if new_tokens <= 0:
        # Nothing new to sample
        return zs
    
    # get z_conds from level above
    z_conds = prior.get_z_conds(zs, start, end)

    # set y offset, sample_length and lyrics tokens
    y = prior.get_y(labels, start)

    empty_cache()

    max_batch_size = sampling_kwargs['max_batch_size']
    del sampling_kwargs['max_batch_size']

    z_list = split_batch(z, n_samples, max_batch_size)
    z_conds_list = split_batch(z_conds, n_samples, max_batch_size)
    y_list = split_batch(y, n_samples, max_batch_size)
    z_samples = []
    for z_i, z_conds_i, y_i in zip(z_list, z_conds_list, y_list):
        z_samples_i = prior.sample(n_samples=z_i.shape[0], z=z_i, z_conds=z_conds_i, y=y_i, **sampling_kwargs)
        z_samples.append(z_samples_i)
    z = t.cat(z_samples, dim=0)

    sampling_kwargs['max_batch_size'] = max_batch_size

    # Update z with new sample
    z_new = z[:,-new_tokens:]
    zs[level] = t.cat([zs[level], z_new], dim=1)
    return zs

# Sample total_length tokens at level=level with hop_length=hop_length
def sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):
    for start in range(0, total_length - prior.n_ctx + hop_length, hop_length):
        zs = sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)
    return zs

# Sample multiple levels
def _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps):
    alignments = None
    for level in reversed(sample_levels):
        prior = priors[level]
        prior.cuda()
        empty_cache()

        # Set correct total_length, hop_length, labels and sampling_kwargs for level
        total_length = hps.sample_length//prior.raw_to_tokens
        hop_length = int(hps.hop_fraction[level]*prior.n_ctx)
        zs = sample_level(zs, labels[level], sampling_kwargs[level], level, prior, total_length, hop_length, hps)

        prior.cpu()
        empty_cache()

        # Decode sample
        x = priors[-1].decode(zs[level:], start_level=level)

        logdir = f"{hps.name}/level_{level}"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        t.save(dict(zs=zs, labels=labels, sampling_kwargs=sampling_kwargs, x=x), f"{logdir}/data.pth.tar")
        save_wav(logdir, x, hps.sr)
        if alignments is None and priors[-1].n_tokens > 0:
            alignments = get_alignment(x, zs, labels[-1], priors[-1], sampling_kwargs[-1]['fp16'], hps)
        save_html(logdir, x, zs, labels[-1], alignments, hps)
    return zs

# Ancestral sample
def ancestral_sample(labels, sampling_kwargs, priors, hps):
    sample_levels = list(range(len(priors)))
    zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
    return zs

# Upsample
def upsample(zs, labels, sampling_kwargs, priors, hps):
    sample_levels = list(range(len(priors) - 1))
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
    return zs

# Primed sample
def primed_sample(x, labels, sampling_kwargs, priors, hps):
    sample_levels = list(range(len(priors)))
    # TODO: Confirm if vqvae encode supports shorter length segments
    zs = priors[-1].encode(x, start_level=0, end_level=len(priors))
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
    return zs

def save_samples(model, device, hps):
    print(hps)
    from jukebox.lyricdict import poems, gpt_2_lyrics
    vqvae, priors = make_model(model, device, hps)

    total_length = 214 * hps.sr
    offset = 0
    metas = [dict(artist = "Alan Jackson",
                genre = "Country",
                lyrics = poems['ozymandias'],
                total_length=total_length,
                offset=offset,
                ),
             dict(artist="Joe Bonamassa",
                  genre="Blues Rock",
                  lyrics=gpt_2_lyrics['hottub'],
                  total_length=total_length,
                  offset=offset,
                  ),
             dict(artist="Frank Sinatra",
                  genre="Classic Pop",
                  lyrics=gpt_2_lyrics['alone'],
                  total_length=total_length,
                  offset=offset,
                  ),
             dict(artist="Ella Fitzgerald",
                  genre="Jazz",
                  lyrics=gpt_2_lyrics['count'],
                  total_length=total_length,
                  offset=offset,
                  ),
             ]
    while len(metas) < hps.n_samples:
        metas.extend(metas)
    metas = metas[:hps.n_samples]

    labels = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in priors]
    for label in labels:
        assert label['y'].shape[0] == hps.n_samples

    lower_level_chunk_size = 32
    lower_level_max_batch_size = 16
    if model == '1b_lyrics':
        chunk_size = 32
        max_batch_size = 16
    else:
        chunk_size = 16
        max_batch_size = 3
    sampling_kwargs = [dict(temp=0.99, fp16=True, chunk_size=lower_level_chunk_size, max_batch_size=lower_level_max_batch_size),
                       dict(temp=0.99, fp16=True, chunk_size=lower_level_chunk_size, max_batch_size=lower_level_max_batch_size),
                       dict(temp=0.99, fp16=True, chunk_size=chunk_size, max_batch_size=max_batch_size)]

    ancestral_sample(labels, sampling_kwargs, priors, hps)


def run(model, port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = Hyperparams(**kwargs)

    with t.no_grad():
        save_samples(model, device, hps)

if __name__ == '__main__':
    fire.Fire(run)
