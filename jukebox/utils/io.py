import numpy as np
import av
import torch as t
import jukebox.utils.dist_adapter as dist

def get_duration_sec(file, cache=False):
    try:
        with open(file + '.dur', 'r') as f:
            duration = float(f.readline().strip('\n'))
        return duration
    except:
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + '.dur', 'w') as f:
                f.write(str(duration) + '\n')
        return duration

def load_audio(file, sr, offset, duration, resample=True, approx=False, time_base='samples', check_duration=True):
    if time_base == 'sec':
        offset = offset * sr
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    container = av.open(file)
    audio = container.streams.get(audio=0)[0] # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    if approx:
        if offset + duration > audio_duration*sr:
            # Move back one window. Cap at audio_duration
            offset = np.min(audio_duration*sr - duration, offset - duration)
    else:
        if check_duration:
            assert offset + duration <= audio_duration*sr, f'End {offset + duration} beyond duration {audio_duration*sr}'
    if resample:
        resampler = av.AudioResampler(format='fltp',layout='stereo', rate=sr)
    else:
        assert sr == audio.sample_rate
    offset = int(offset / sr / float(audio.time_base)) #int(offset / float(audio.time_base)) # Use units of time_base for seeking
    duration = int(duration) #duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    for frame in container.decode(audio=0): # Only first audio stream
        if resample:
            frame.pts = None
            frame = resampler.resample(frame)
        frame = frame.to_ndarray(format='fltp') # Convert to floats and not int16
        read = frame.shape[-1]
        if total_read + read > duration:
            read = duration - total_read
        sig[:, total_read:total_read + read] = frame[:, :read]
        total_read += read
        if total_read == duration:
            break
    assert total_read <= duration, f'Expected {duration} frames, got {total_read}'
    return sig, sr

def test_simple_loader():
    import librosa
    from tqdm import tqdm

    collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], dim=0)

    def get_batch(file, loader):
        y1, sr = loader(file, sr=44100, offset=0.0, duration=6.0, time_base='sec')
        y2, sr = loader(file, sr=44100, offset=20.0, duration=6.0, time_base='sec')
        return [y1, y2]

    def load(file, loader):
        batch = get_batch(file, loader)  # np
        x = collate_fn(batch)  # torch cpu
        x = x.to('cuda', non_blocking=True)  # torch gpu
        return x

    files = librosa.util.find_files('/root/data/', ['mp3', 'm4a', 'opus'])
    print(files[:10])
    loader = load_audio
    print("Loader", loader.__name__)
    x = t.randn(2, 2).cuda()
    x = load(files[0], loader)
    for i,file in enumerate(tqdm(files)):
        x = load(file, loader)
        if i == 100:
            break

def test_dataset_loader():
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from jukebox.utils.audio_utils import audio_preprocess, audio_postprocess
    from jukebox.hparams import setup_hparams
    from jukebox.data.files_dataset import FilesAudioDataset
    hps = setup_hparams("teeny", {})
    hps.sr = 22050  # 44100
    hps.hop_length = 512
    hps.labels = False
    hps.channels = 2
    hps.aug_shift = False
    hps.bs = 2
    hps.nworkers = 2 # Getting 20 it/s with 2 workers, 10 it/s with 1 worker
    print(hps)
    dataset = hps.dataset
    root = hps.root
    from tensorboardX import SummaryWriter
    sr = {22050: '22k', 44100: '44k', 48000: '48k'}[hps.sr]
    writer = SummaryWriter(f'{root}/{dataset}/logs/{sr}/logs')
    dataset = FilesAudioDataset(hps)
    print("Length of dataset", len(dataset))

    # Torch Loader
    collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], 0)
    sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=hps.bs, num_workers=hps.nworkers, pin_memory=False, sampler=sampler,
                              drop_last=True, collate_fn=collate_fn)

    dist.barrier()
    sampler.set_epoch(0)
    for i, x in enumerate(tqdm(train_loader)):
        x = x.to('cuda', non_blocking=True)
        for j, aud in enumerate(x):
            writer.add_audio('in_' + str(i*hps.bs + j), aud, 1, hps.sr)
        print("Wrote in")
        x = audio_preprocess(x, hps)
        x = audio_postprocess(x, hps)
        for j, aud in enumerate(x):
            writer.add_audio('out_' + str(i*hps.bs + j), aud, 1, hps.sr)
        print("Wrote out")
        dist.barrier()
        break

if __name__ == '__main__':
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    setup_dist_from_mpi(port=29500)
    test_dataset_loader()

