"""
Make model classes
Load from checkpoints
Test on dummy outputs to see if everything matches
"""
import os
import numpy as np
import torch as t
import jukebox.utils.dist_adapter as dist
from jukebox.hparams import Hyperparams, setup_hparams, REMOTE_PREFIX
from jukebox.utils.remote_utils import download
from jukebox.utils.torch_utils import freeze_model
from jukebox.utils.dist_utils import print_all
from jukebox.vqvae.vqvae import calculate_strides
import fire

MODELS = {
    '5b': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_5b"),
    '5b_lyrics': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_5b_lyrics"),
    '1b_lyrics': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_1b_lyrics"),
    #'your_model': ("you_vqvae_here", "your_upsampler_here", ..., "you_top_level_prior_here")
}

def load_checkpoint(path):
    restore = path
    if restore.startswith(REMOTE_PREFIX):
        remote_path = restore
        local_path = os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):])
        if dist.get_rank() % 8 == 0:
            print("Downloading from azure")
            if not os.path.exists(os.path.dirname(local_path)):
                os.makedirs(os.path.dirname(local_path))
            if not os.path.exists(local_path):
                download(remote_path, local_path)
        restore = local_path
    dist.barrier()
    checkpoint = t.load(restore, map_location=t.device('cpu'))
    print("Restored from {}".format(restore))
    return checkpoint

def save_checkpoint(logger, name, model, opt, metrics, hps):
    with t.no_grad():
        save_hps = {**hps}
        save_hps = {k: v for k,v in save_hps.items() if k not in ['metadata_v2','metadata_v3', 'alignments', 'lyric_processor', 'midi_processor']}
        t.save({'hps': save_hps,
                'model': model.state_dict(), # should also save bottleneck k's as buffers
                'opt': opt.state_dict() if opt is not None else None,
                'step': logger.iters,
                **metrics}, f'{logger.logdir}/checkpoint_{name}.pth.tar')
    return

def restore_model(hps, model, checkpoint_path):
    model.step = 0
    if checkpoint_path != '':
        checkpoint = load_checkpoint(checkpoint_path)
        # checkpoint_hps = Hyperparams(**checkpoint['hps'])
        # for k in set(checkpoint_hps.keys()).union(set(hps.keys())):
        #     if checkpoint_hps.get(k, None) != hps.get(k, None):
        #         print(k, "Checkpoint:", checkpoint_hps.get(k, None), "Ours:", hps.get(k, None))
        checkpoint['model'] = {k[7:] if k[:7] == 'module.' else k: v for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'])
        if 'step' in checkpoint: model.step = checkpoint['step']

def restore_opt(opt, shd, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = load_checkpoint(checkpoint_path)
    if "opt" in checkpoint:
        opt.load_state_dict(checkpoint['opt'])
    if "step" in checkpoint:
        shd.step(checkpoint['step'])

def make_vqvae(hps, device='cuda'):
    from jukebox.vqvae.vqvae import VQVAE
    block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv,
                        dilation_growth_rate=hps.dilation_growth_rate,
                        dilation_cycle=hps.dilation_cycle,
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

    if not hps.sample_length:
        assert hps.sample_length_in_seconds != 0
        downsamples = calculate_strides(hps.strides_t, hps.downs_t)
        top_raw_to_tokens = np.prod(downsamples)
        hps.sample_length = (hps.sample_length_in_seconds * hps.sr // top_raw_to_tokens) * top_raw_to_tokens
        print(f"Setting sample length to {hps.sample_length} (i.e. {hps.sample_length/hps.sr} seconds) to be multiple of {top_raw_to_tokens}")

    vqvae = VQVAE(input_shape=(hps.sample_length,1), levels=hps.levels, downs_t=hps.downs_t, strides_t=hps.strides_t,
                  emb_width=hps.emb_width, l_bins=hps.l_bins,
                  mu=hps.l_mu, commit=hps.commit,
                  spectral=hps.spectral, multispectral=hps.multispectral,
                  multipliers=hps.hvqvae_multipliers, use_bottleneck=hps.use_bottleneck,
                  **block_kwargs)

    vqvae = vqvae.to(device)
    restore_model(hps, vqvae, hps.restore_vqvae)
    if hps.train and not hps.prior:
        print_all(f"Loading vqvae in train mode")
        if hps.restore_vqvae != '':
            print_all("Reseting bottleneck emas")
            for level, bottleneck in enumerate(vqvae.bottleneck.level_blocks):
                num_samples = hps.sample_length
                downsamples = calculate_strides(hps.strides_t, hps.downs_t)
                raw_to_tokens = np.prod(downsamples[:level + 1])
                num_tokens = (num_samples // raw_to_tokens) * dist.get_world_size()
                bottleneck.restore_k(num_tokens=num_tokens, threshold=hps.revival_threshold)
    else:
        print_all(f"Loading vqvae in eval mode")
        vqvae.eval()
        freeze_model(vqvae)
    return vqvae

def make_prior(hps, vqvae, device='cuda'):
    from jukebox.prior.prior import SimplePrior

    prior_kwargs = dict(input_shape=(hps.n_ctx,), bins=vqvae.l_bins,
                        width=hps.prior_width, depth=hps.prior_depth, heads=hps.heads,
                        attn_order=hps.attn_order, blocks=hps.blocks, spread=hps.spread,
                        attn_dropout=hps.attn_dropout, resid_dropout=hps.resid_dropout, emb_dropout=hps.emb_dropout,
                        zero_out=hps.zero_out, res_scale=hps.res_scale, pos_init=hps.pos_init,
                        init_scale=hps.init_scale,
                        m_attn=hps.m_attn, m_mlp=hps.m_mlp,
                        checkpoint_res=hps.c_res if hps.train else 0, checkpoint_attn=hps.c_attn if hps.train else 0, checkpoint_mlp=hps.c_mlp if hps.train else 0)

    x_cond_kwargs = dict(out_width=hps.prior_width, init_scale=hps.init_scale,
                         width=hps.cond_width, depth=hps.cond_depth, m_conv=hps.cond_m_conv,
                         dilation_growth_rate=hps.cond_dilation_growth_rate, dilation_cycle=hps.cond_dilation_cycle,
                         zero_out=hps.cond_zero_out, res_scale=hps.cond_res_scale,
                         checkpoint_res=hps.cond_c_res)  # have to keep this else names wrong

    y_cond_kwargs = dict(out_width=hps.prior_width, init_scale=hps.init_scale,
                         y_bins=hps.y_bins, t_bins=hps.t_bins, sr= hps.sr, min_duration=hps.min_duration,
                         max_duration=hps.max_duration, max_bow_genre_size=hps.max_bow_genre_size)

    if hps.use_tokens and not hps.single_enc_dec:
        prime_kwargs = dict(use_tokens=hps.use_tokens, prime_loss_fraction=hps.prime_loss_fraction,
                            n_tokens=hps.n_tokens, bins=hps.n_vocab,
                            width=hps.prime_width, depth=hps.prime_depth, heads=hps.prime_heads,
                            attn_order=hps.prime_attn_order, blocks=hps.prime_blocks, spread=hps.prime_spread,
                            attn_dropout=hps.prime_attn_dropout, resid_dropout=hps.prime_resid_dropout,
                            emb_dropout=hps.prime_emb_dropout,
                            zero_out=hps.prime_zero_out, res_scale=hps.prime_res_scale,
                            pos_init=hps.prime_pos_init, init_scale=hps.prime_init_scale,
                            m_attn=hps.prime_m_attn, m_mlp=hps.prime_m_mlp,
                            checkpoint_res=hps.prime_c_res if hps.train else 0, checkpoint_attn=hps.prime_c_attn if hps.train else 0,
                            checkpoint_mlp=hps.prime_c_mlp if hps.train else 0)
    else:
        prime_kwargs = dict(use_tokens=hps.use_tokens, prime_loss_fraction=hps.prime_loss_fraction,
                            n_tokens=hps.n_tokens, bins=hps.n_vocab)

    # z_shapes for other levels given this level gets n_ctx codes
    rescale = lambda z_shape: (z_shape[0]*hps.n_ctx//vqvae.z_shapes[hps.level][0],)
    z_shapes = [rescale(z_shape) for z_shape in vqvae.z_shapes]

    prior = SimplePrior(z_shapes=z_shapes,
                        l_bins=vqvae.l_bins,
                        encoder=vqvae.encode,
                        decoder=vqvae.decode,
                        level=hps.level,
                        downs_t=vqvae.downs_t,
                        strides_t=vqvae.strides_t,
                        labels=hps.labels,
                        prior_kwargs=prior_kwargs,
                        x_cond_kwargs=x_cond_kwargs,
                        y_cond_kwargs=y_cond_kwargs,
                        prime_kwargs=prime_kwargs,
                        copy_input=hps.copy_input,
                        labels_v3=hps.labels_v3,
                        merged_decoder=hps.merged_decoder,
                        single_enc_dec=hps.single_enc_dec)

    prior.alignment_head = hps.get('alignment_head', None)
    prior.alignment_layer = hps.get('alignment_layer', None)

    if hps.fp16_params:
        print_all("Converting to fp16 params")
        from jukebox.transformer.ops import _convert_conv_weights_to_fp16
        prior.apply(_convert_conv_weights_to_fp16)
    prior = prior.to(device)
    restore_model(hps, prior, hps.restore_prior)
    if hps.train:
        print_all(f"Loading prior in train mode")
        pass
    else:
        print_all(f"Loading prior in eval mode")
        prior.eval()
        freeze_model(prior)
    return prior

def make_model(model, device, hps, levels=None):
    vqvae, *priors = MODELS[model]
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=hps.get('sample_length', 0), sample_length_in_seconds=hps.get('sample_length_in_seconds', 0))), device)
    hps.sample_length = vqvae.sample_length
    if levels is None:
        levels = range(len(priors))
    priors = [make_prior(setup_hparams(priors[level], dict()), vqvae, 'cpu') for level in levels]
    return vqvae, priors

def save_outputs(model, device, hps):
    # Check logits
    if hps.labels_v3:
        n_ctx = 6144
        n_tokens = 384
        prime_bins = 79
    else:
        n_ctx = 8192
        n_tokens = 512
        prime_bins = 80

    rng = t.random.manual_seed(0)
    x = 2 * t.rand((1, n_ctx * 8 * 4 * 4, 1), generator=rng, dtype=t.float).cuda() - 1.0  # -1 to 1
    lyric_tokens = t.randint(0, prime_bins, (1, n_tokens), generator=rng, dtype=t.long).view(-1).numpy()
    artist_id = 10
    genre_ids = [1]
    total_length = 2 * 2646000
    offset = 2646000

    vqvae, priors = make_model(model, device, hps)

    # encode
    vq_prior = priors[-1]
    zs = vq_prior.encode(x, start_level=0)
    x_ds = [vq_prior.decode(zs[level:], start_level=level) for level in range(0, len(zs))]

    # priors
    data = dict(zs=zs, x_ds=x_ds)
    for level in range(len(priors)):
        print(f"Doing level {level}")
        if hps.labels_v3 and level != hps.levels - 1:
            print(f"Skipping level {level}")
            continue
        prior = priors[level]
        prior.cuda()
        x_in = x[:, :n_ctx * 8 * (4 ** level)]
        y_in = t.from_numpy(prior.labeller.get_y_from_ids(artist_id, genre_ids, lyric_tokens, total_length, offset)).view(1, -1).cuda().long()
        x_out, _, metrics = prior(x_in, y_in, fp16=hps.fp16, get_preds=True, decode=True)
        preds = metrics['preds']
        data[level] = dict(x=x_in, y=y_in, x_out=x_out, preds=preds)
        prior.cpu()
    t.save(data, 'data.pth.tar')
    dist.barrier()
    print("Saved data")
    exit()


def run(model, port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = Hyperparams(**kwargs)

    with t.no_grad():
        save_outputs(model, device, hps)

if __name__ == '__main__':
    fire.Fire(run)
