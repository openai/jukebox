import fire
import os
import torch

from jukebox.hparams import Hyperparams, DEFAULTS
from jukebox.utils.audio_utils import save_wav, load_audio
from jukebox.make_models import make_vqvae


def run(
    audio_file=None,
    sr=44100,
    offset=0,
    duration_in_sec=6,
    mono=True,
    port=29500,
    **kwargs,
):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)

    for default in ["vqvae", "vqvae_conv_block", "train_test_eval"]:
        for k, v in DEFAULTS[default].items():
            kwargs.setdefault(k, v)

    hps = Hyperparams(**kwargs)
    inp = load_audio(audio_file, sr=sr, offset=offset, duration=duration_in_sec * sr, mono=mono)

    hps.sample_length = inp.shape[1]
    vqvae = make_vqvae(hps)

    with torch.no_grad():
        inp = torch.tensor(inp, device='cuda').T.unsqueeze(0)
        zs = vqvae.encode(inp)
        decoded = vqvae.decode(zs)

    save_wav(".", decoded, sr)

if __name__ == '__main__':
    fire.Fire(run)

