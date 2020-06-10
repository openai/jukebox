import os
import torch as t
import jukebox.utils.dist_adapter as dist

from jukebox.hparams import Hyperparams
from jukebox.data.labels import EmptyLabeller
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav, load_audio
from jukebox.make_models import make_model
from jukebox.align import get_alignment
from jukebox.save_html import save_html
from jukebox.utils.sample_utils import split_batch, get_starts
from jukebox.utils.dist_utils import print_once
import fire

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

    print_once(f"Sampling {sample_tokens} tokens for [{start},{start+sample_tokens}]. Conditioning on {conditioning_tokens} tokens")

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
    print_once(f"Sampling level {level}")
    if total_length >= prior.n_ctx:
        for start in get_starts(total_length, prior.n_ctx, hop_length):
            zs = sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)
    else:
        zs = sample_partial_window(zs, labels, sampling_kwargs, level, prior, total_length, hps)
    return zs

# Sample multiple levels
def _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps):
    alignments = None
    for level in reversed(sample_levels):
        prior = priors[level]
        prior.cuda()
        empty_cache()

        # Set correct total_length, hop_length, labels and sampling_kwargs for level
        assert hps.sample_length % prior.raw_to_tokens == 0, f"Expected sample_length {hps.sample_length} to be multiple of {prior.raw_to_tokens}"
        total_length = hps.sample_length//prior.raw_to_tokens
        hop_length = int(hps.hop_fraction[level]*prior.n_ctx)
        zs = sample_level(zs, labels[level], sampling_kwargs[level], level, prior, total_length, hop_length, hps)

        prior.cpu()
        empty_cache()

        # Decode sample
        x = prior.decode(zs[level:], start_level=level, bs_chunks=zs[level].shape[0])

        if dist.get_world_size() > 1:
            logdir = f"{hps.name}_rank_{dist.get_rank()}/level_{level}"
        else:
            logdir = f"{hps.name}/level_{level}"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        t.save(dict(zs=zs, labels=labels, sampling_kwargs=sampling_kwargs, x=x), f"{logdir}/data.pth.tar")
        save_wav(logdir, x, hps.sr)
        if alignments is None and priors[-1] is not None and priors[-1].n_tokens > 0 and not isinstance(priors[-1].labeller, EmptyLabeller):
            alignments = get_alignment(x, zs, labels[-1], priors[-1], sampling_kwargs[-1]['fp16'], hps)
        save_html(logdir, x, zs, labels[-1], alignments, hps)
    return zs

# Generate ancestral samples given a list of artists and genres
def ancestral_sample(labels, sampling_kwargs, priors, hps):
    sample_levels = list(range(len(priors)))
    zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
    return zs

# Continue ancestral sampling from previously saved codes
def continue_sample(zs, labels, sampling_kwargs, priors, hps):
    sample_levels = list(range(len(priors)))
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
    return zs

# Upsample given already generated upper-level codes
def upsample(zs, labels, sampling_kwargs, priors, hps):
    sample_levels = list(range(len(priors) - 1))
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
    return zs

# Prompt the model with raw audio input (dimension: NTC) and generate continuations
def primed_sample(x, labels, sampling_kwargs, priors, hps):
    sample_levels = list(range(len(priors)))
    zs = priors[-1].encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps)
    return zs

# Load `duration` seconds of the given audio files to use as prompts
def load_prompts(audio_files, duration, hps):
    xs = []
    for audio_file in audio_files:
        x = load_audio(audio_file, sr=hps.sr, duration=duration, offset=0.0, mono=True)
        x = x.T # CT -> TC
        xs.append(x)
    while len(xs) < hps.n_samples:
        xs.extend(xs)
    xs = xs[:hps.n_samples]
    x = t.stack([t.from_numpy(x) for x in xs])
    x = x.to('cuda', non_blocking=True)
    return x

# Load codes from previous sampling run
def load_codes(codes_file, duration, priors, hps):
    data = t.load(codes_file, map_location='cpu')
    zs = [z.cuda() for z in data['zs']]
    assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
    del data
    if duration is not None:
        # Cut off codes to match duration
        top_raw_to_tokens = priors[-1].raw_to_tokens
        assert duration % top_raw_to_tokens == 0, f"Cut-off duration {duration} not an exact multiple of top_raw_to_tokens"
        assert duration//top_raw_to_tokens <= zs[-1].shape[1], f"Cut-off tokens {duration//priors[-1].raw_to_tokens} longer than tokens {zs[-1].shape[1]} in saved codes"
        zs = [z[:,:duration//prior.raw_to_tokens] for z, prior in zip(zs, priors)]
    return zs

# Generate and save samples, alignment, and webpage for visualization.
def save_samples(model, device, hps, sample_hps):
    print(hps)
    from jukebox.lyricdict import poems, gpt_2_lyrics
    vqvae, priors = make_model(model, device, hps)

    assert hps.sample_length//priors[-2].raw_to_tokens >= priors[-2].n_ctx, f"Upsampling needs atleast one ctx in get_z_conds. Please choose a longer sample length"

    total_length = hps.total_sample_length_in_seconds * hps.sr
    offset = 0

    # Set artist/genre/lyrics for your samples here!
    # We used different label sets in our models, but you can write the human friendly names here and we'll map them under the hood for each model.
    # For the 5b/5b_lyrics model and the upsamplers, labeller will look up artist and genres in v2 set. (after lowercasing, removing non-alphanumerics and collapsing whitespaces to _).
    # For the 1b_lyrics top level, labeller will look up artist and genres in v3 set (after lowercasing).
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
             dict(artist="Céline Dion",
                  genre="Pop",
                  lyrics=gpt_2_lyrics['darkness'],
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

    if sample_hps.mode == 'ancestral':
        ancestral_sample(labels, sampling_kwargs, priors, hps)
    elif sample_hps.mode in ['continue', 'upsample']:
        assert sample_hps.codes_file is not None
        top_raw_to_tokens = priors[-1].raw_to_tokens
        if sample_hps.prompt_length_in_seconds is not None:
            duration = (int(sample_hps.prompt_length_in_seconds * hps.sr) // top_raw_to_tokens) * top_raw_to_tokens
        else:
            duration = None
        zs = load_codes(sample_hps.codes_file, duration, priors, hps)
        if sample_hps.mode == 'continue':
            continue_sample(zs, labels, sampling_kwargs, priors, hps)
        elif sample_hps.mode == 'upsample':
            upsample(zs, labels, sampling_kwargs, priors, hps)
    elif sample_hps.mode == 'primed':
        assert sample_hps.audio_file is not None
        assert sample_hps.prompt_length_in_seconds is not None
        audio_files = sample_hps.audio_file.split(',')
        top_raw_to_tokens = priors[-1].raw_to_tokens
        duration = (int(sample_hps.prompt_length_in_seconds * hps.sr) // top_raw_to_tokens) * top_raw_to_tokens
        x = load_prompts(audio_files, duration, hps)
        primed_sample(x, labels, sampling_kwargs, priors, hps)
    else:
        raise ValueError(f'Unknown sample mode {sample_hps.mode}.')


def run(model, mode='ancestral', codes_file=None, audio_file=None, prompt_length_in_seconds=None, port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = Hyperparams(**kwargs)
    sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

    with t.no_grad():
        save_samples(model, device, hps, sample_hps)

if __name__ == '__main__':
    fire.Fire(run)
