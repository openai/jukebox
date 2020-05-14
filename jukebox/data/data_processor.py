import torch as t
import jukebox.utils.dist_adapter as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from jukebox.utils.dist_utils import print_all
from jukebox.utils.audio_utils import calculate_bandwidth
from jukebox.data.files_dataset import FilesAudioDataset

class OffsetDataset(Dataset):
    def __init__(self, dataset, start, end, test=False):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end
        self.test = test
        assert 0 <= self.start < self.end <= len(self.dataset)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        return self.dataset.get_item(self.start + item, test=self.test)

class DataProcessor():
    def __init__(self, hps):
        self.dataset = FilesAudioDataset(hps)
        duration = 1 if hps.prior else 600
        hps.bandwidth = calculate_bandwidth(self.dataset, hps, duration=duration)
        self.create_datasets(hps)
        self.create_samplers(hps)
        self.create_data_loaders(hps)
        self.print_stats(hps)

    def set_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.test_sampler.set_epoch(epoch)

    def create_datasets(self, hps):
        train_len = int(len(self.dataset) * hps.train_test_split)
        self.train_dataset = OffsetDataset(self.dataset, 0, train_len, test=False)
        self.test_dataset = OffsetDataset(self.dataset, train_len, len(self.dataset), test=True)

    def create_samplers(self, hps):
        if not dist.is_available():
            self.train_sampler = BatchSampler(RandomSampler(self.train_dataset), batch_size=hps.bs, drop_last=True)
            self.test_sampler = BatchSampler(RandomSampler(self.test_dataset), batch_size=hps.bs, drop_last=True)
        else:
            self.train_sampler = DistributedSampler(self.train_dataset)
            self.test_sampler = DistributedSampler(self.test_dataset)

    def create_data_loaders(self, hps):
        # Loader to load mini-batches
        if hps.labels:
            collate_fn = lambda batch: tuple(t.stack([t.from_numpy(b[i]) for b in batch], 0) for i in range(2))
        else:
            collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], 0)

        print('Creating Data Loader')
        self.train_loader = DataLoader(self.train_dataset, batch_size=hps.bs, num_workers=hps.nworkers,
                                       sampler=self.train_sampler, pin_memory=False,
                                       drop_last=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=hps.bs, num_workers=hps.nworkers,
                                      sampler=self.test_sampler, pin_memory=False,
                                      drop_last=False, collate_fn=collate_fn)

    def print_stats(self, hps):
        print_all(f"Train {len(self.train_dataset)} samples. Test {len(self.test_dataset)} samples")
        print_all(f'Train sampler: {self.train_sampler}')
        print_all(f'Train loader: {len(self.train_loader)}')
