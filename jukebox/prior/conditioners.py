import torch as t
import torch.nn as nn

from jukebox.transformer.ops import LayerNorm
from jukebox.vqvae.encdec import DecoderConvBock
from jukebox.utils.torch_utils import assert_shape

class Conditioner(nn.Module):
    def __init__(self, input_shape, bins, down_t, stride_t, out_width, init_scale, zero_out, res_scale, **block_kwargs):
        super().__init__()
        self.x_shape = input_shape

        # Embedding
        self.width = out_width
        self.x_emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)

        # Conditioner
        self.cond = DecoderConvBock(self.width, self.width, down_t, stride_t, **block_kwargs, zero_out=zero_out, res_scale=res_scale)
        self.ln = LayerNorm(self.width)

    def preprocess(self, x):
        x = x.permute(0,2,1) # NTC -> NCT
        return x

    def postprocess(self, x):
        x = x.permute(0,2,1) # NCT -> NTC
        return x

    def forward(self, x, x_cond=None):
        N = x.shape[0]
        assert_shape(x, (N, *self.x_shape))
        if x_cond is not None:
            assert_shape(x_cond, (N, *self.x_shape, self.width))
        else:
            x_cond = 0.0
        # Embed x
        x = x.long()
        x = self.x_emb(x)
        assert_shape(x, (N, *self.x_shape, self.width))
        x = x + x_cond

        # Run conditioner
        x = self.preprocess(x)
        x = self.cond(x)
        x = self.postprocess(x)
        x = self.ln(x)
        return x

def flip(x):
    def _flip(x):
        return x.permute(0,2,1).contiguous()
    if isinstance(x, (list, tuple)):
        return [flip(z) for z in x]
    return _flip(x)

class SimpleEmbedding(nn.Module):
    def __init__(self, bins, out_width, init_scale):
        super().__init__()
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)

    def forward(self, y):
        assert len(y.shape) == 2, f"Expected shape with 2 dims, got {y.shape}"
        assert isinstance(y, t.cuda.LongTensor), f"Expected dtype {t.cuda.LongTensor}, got {y.dtype}"
        assert (0 <= y).all() and (y < self.bins).all(), f"Bins {self.bins}, got label {y}"
        return self.emb(y)

class RangeEmbedding(nn.Module):
    # Interpolating
    # Interpolate so that [pos_start, pos_end] <-> position tensor of length n_ctx
    #
    # Binning
    # For each pos in position tensor, find its bin
    # [start,end) mapped to [0,1,...,bins-1]
    # [start,end) -> [0,1) -> [0, bins) -> floor -> [0,...,bins-1]
    # NOTE: Open ended interval on right, so start <= pos < end, not <= end
    def __init__(self, n_time, bins, range, out_width, init_scale, clamp=False):
        super().__init__()
        self.n_time = n_time
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)
        self.pos_min, self.pos_max = range
        self.clamp = clamp

    def forward(self, pos_start, pos_end=None):
        # Check if [pos_start,pos_end] in [pos_min, pos_max)
        assert len(pos_start.shape) == 2, f"Expected shape with 2 dims, got {pos_start.shape}"
        assert (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all(), f"Range is [{self.pos_min},{self.pos_max}), got {pos_start}"
        pos_start = pos_start.float()
        if pos_end is not None:
            assert len(pos_end.shape) == 2, f"Expected shape with 2 dims, got {pos_end.shape}"
            if self.clamp:
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)
            assert (self.pos_min <= pos_end).all() and (pos_end <= self.pos_max).all(), f"Range is [{self.pos_min},{self.pos_max}), got {pos_end}"
            pos_end = pos_end.float()
        # Interpolate so that [pos_start, ..., pos_end] <-> position tensor of length n_ctx
        n_time = self.n_time
        if n_time != 1:
            assert pos_end is not None
            interpolation  = (t.arange(0, n_time, dtype=t.float, device='cuda').view(1,n_time)/n_time)
            position = pos_start + (pos_end - pos_start)*interpolation
        else:
            position = pos_start

        # Bin each value to bins
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min) # [0,1)
        bins = (self.bins * normalised_position).floor().long().detach() # [0,1) -> [0,1..,bins) -> [0,1...,bins-1]
        return self.emb(bins)

class LabelConditioner(nn.Module):
    def __init__(self, y_bins, t_bins, sr, min_duration, max_duration, n_time, out_width, init_scale, max_bow_genre_size, include_time_signal):
        super().__init__()
        self.n_time = n_time
        self.out_width = out_width
        assert len(y_bins) == 2, f"Expecting (genre, artist) bins, got {y_bins}"
        bow_genre_bins, artist_bins = y_bins
        self.max_bow_genre_size = max_bow_genre_size
        self.bow_genre_emb = SimpleEmbedding(bow_genre_bins, out_width, init_scale)
        self.artist_emb = SimpleEmbedding(artist_bins, out_width, init_scale)
        self.include_time_signal = include_time_signal
        if self.include_time_signal:
            t_ranges = ((min_duration * sr, max_duration * sr),  # Total length
                        (0.0, max_duration * sr),                # Absolute pos
                        (0.0, 1.0))                              # Relative pos
            assert len(t_ranges) == 3, f"Expecting (total, absolute, relative) ranges, got {t_ranges}"
            total_length_range, absolute_pos_range, relative_pos_range = t_ranges
            self.total_length_emb = RangeEmbedding(1, t_bins, total_length_range, out_width, init_scale)
            self.absolute_pos_emb = RangeEmbedding(n_time, t_bins, absolute_pos_range, out_width, init_scale)
            self.relative_pos_emb = RangeEmbedding(n_time, t_bins, relative_pos_range, out_width, init_scale, clamp=True)

    def forward(self, y):
        assert len(y.shape) == 2, f"Expected shape with 2 dims, got {y.shape}"
        assert y.shape[-1] == 4 + self.max_bow_genre_size, f"Expected shape (N,{4 + self.max_bow_genre_size}), got {y.shape}"
        assert isinstance(y, t.cuda.LongTensor), f"Expected dtype {t.cuda.LongTensor}, got {y.dtype}"
        N = y.shape[0]
        total_length, offset, length, artist, genre = y[:,0:1], y[:,1:2], y[:,2:3], y[:,3:4], y[:,4:]

        # Start embedding of length 1
        artist_emb = self.artist_emb(artist)
        # Empty genre slots are denoted by -1. We mask these out.
        mask = (genre >= 0).float().unsqueeze(2)
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)
        start_emb = genre_emb + artist_emb
        assert_shape(start_emb, (N, 1, self.out_width))

        # Pos embedding of length n_ctx
        if self.include_time_signal:
            start, end = offset, offset + length
            total_length, start, end = total_length.float(), start.float(), end.float()
            pos_emb = self.total_length_emb(total_length) + self.absolute_pos_emb(start, end) + self.relative_pos_emb(start/total_length, end/total_length)
            assert_shape(pos_emb, (N, self.n_time, self.out_width))
        else:
            pos_emb = None
        return start_emb, pos_emb