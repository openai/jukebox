import numpy as np
import torch as t
import torch.nn as nn
import jukebox.utils.dist_adapter as dist

from jukebox.transformer.ops import LayerNorm
from jukebox.prior.autoregressive import ConditionalAutoregressive2D
from jukebox.prior.conditioners import Conditioner, LabelConditioner
from jukebox.data.labels import EmptyLabeller, Labeller

from jukebox.utils.torch_utils import assert_shape
from jukebox.utils.dist_utils import print_once
from jukebox.vqvae.vqvae import calculate_strides


"""
Model the prior on vq codes conditioned on timing, artist, genre, lyrics and codes from levels above. 
To condition on the timing, genre and artist, we use the LabelConditioner class
To condition on the codes from the level above, we use the Conditioner class
To condition on lyrics, we allow two types of priors:
- Separate Encoder Decoder: This is the usual encoder-decoder style transformer. The encoder transformer autoregressively 
models the lyrics, and we use its last layer to produce keys/values that are attened to by the decoder transformer
- Single Encoder Decoder: This is a simplification where we combine them into a single model. We merge the text vocab 
and VQ vocab into a single large vocab, and the lyric tokens and VQ tokens into a single longer sequence of tokens which 
we autoregressively model together.
"""
class SimplePrior(nn.Module):
    def __init__(self, z_shapes, l_bins, encoder, decoder, level,
                 downs_t, strides_t, labels, prior_kwargs, x_cond_kwargs, y_cond_kwargs,
                 prime_kwargs, copy_input, labels_v3=False,
                 merged_decoder=False, single_enc_dec=False):
        super().__init__()

        self.use_tokens = prime_kwargs.pop('use_tokens')
        self.n_tokens = prime_kwargs.pop('n_tokens')
        self.prime_loss_fraction = prime_kwargs.pop('prime_loss_fraction')

        self.copy_input = copy_input
        if self.copy_input:
            prime_kwargs['bins'] = l_bins

        self.z_shapes = z_shapes
        self.levels = len(self.z_shapes)

        self.z_shape = self.z_shapes[level]

        self.level = level
        assert level < self.levels, f"Total levels {self.levels}, got level {level}"

        self.l_bins = l_bins

        # Passing functions instead of the vqvae module to avoid getting params
        self.encoder = encoder
        self.decoder = decoder

        # X conditioning
        self.x_cond = (level != (self.levels - 1))
        self.cond_level = level + 1

        # Y conditioning
        self.y_cond = labels

        self.single_enc_dec = single_enc_dec
        # X conditioning
        if self.x_cond:
            self.conditioner_blocks = nn.ModuleList()
            conditioner_block = lambda _level: Conditioner(input_shape=z_shapes[_level],
                                                          bins=l_bins,
                                                          down_t=downs_t[_level],
                                                          stride_t=strides_t[_level],
                                                          **x_cond_kwargs)
            if dist.get_rank() == 0: print(f"Conditioning on 1 above level(s)")
            self.conditioner_blocks.append(conditioner_block(self.cond_level))

        # Y conditioning
        if self.y_cond:
            self.n_time = self.z_shape[0] # Assuming STFT=TF order and raw=T1 order, so T is first dim
            self.y_emb = LabelConditioner(n_time=self.n_time,include_time_signal=not self.x_cond,**y_cond_kwargs)

        # Lyric conditioning
        if single_enc_dec:
            # Single encoder-decoder transformer
            self.prior_shapes = [(self.n_tokens,), prior_kwargs.pop('input_shape')]
            self.prior_bins = [prime_kwargs['bins'], prior_kwargs.pop('bins')]
            self.prior_dims = [np.prod(shape) for shape in self.prior_shapes]
            self.prior_bins_shift = np.cumsum([0, *self.prior_bins])[:-1]
            self.prior_width = prior_kwargs['width']
            print_once(f'Creating cond. autoregress with prior bins {self.prior_bins}, ')
            print_once(f'dims {self.prior_dims}, ')
            print_once(f'shift {self.prior_bins_shift}')
            print_once(f'input shape {sum(self.prior_dims)}')
            print_once(f'input bins {sum(self.prior_bins)}')
            print_once(f'Self copy is {self.copy_input}')

            self.prime_loss_dims, self.gen_loss_dims = self.prior_dims[0], self.prior_dims[1]
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(input_shape=(sum(self.prior_dims),),
                                                     bins=sum(self.prior_bins),
                                                     x_cond=(self.x_cond or self.y_cond), y_cond=True,
                                                     prime_len=self.prime_loss_dims,
                                                     **prior_kwargs)

        else:
            # Separate encoder-decoder transformer
            if self.n_tokens != 0 and self.use_tokens:
                from jukebox.transformer.ops import Conv1D
                prime_input_shape = (self.n_tokens,)
                self.prime_loss_dims = np.prod(prime_input_shape)
                self.prime_acts_width, self.prime_state_width = prime_kwargs['width'], prior_kwargs['width']
                self.prime_prior = ConditionalAutoregressive2D(input_shape=prime_input_shape, x_cond=False, y_cond=False,
                                                               only_encode=True,
                                                               **prime_kwargs)
                self.prime_state_proj = Conv1D(self.prime_acts_width, self.prime_state_width, init_scale=prime_kwargs['init_scale'])
                self.prime_state_ln = LayerNorm(self.prime_state_width)
                self.prime_bins = prime_kwargs['bins']
                self.prime_x_out = nn.Linear(self.prime_state_width, self.prime_bins, bias=False)
                nn.init.normal_(self.prime_x_out.weight, std=0.02 * prior_kwargs['init_scale'])
            else:
                self.prime_loss_dims = 0
            self.gen_loss_dims = np.prod(self.z_shape)
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(x_cond=(self.x_cond or self.y_cond), y_cond=self.y_cond,
                                                     encoder_dims = self.prime_loss_dims, merged_decoder=merged_decoder,
                                                     **prior_kwargs)

        self.n_ctx = self.gen_loss_dims
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.cond_downsample = self.downsamples[level+1] if level != self.levels - 1 else None
        self.raw_to_tokens = np.prod(self.downsamples[:level+1])
        self.sample_length = self.n_ctx*self.raw_to_tokens
        if labels:
            self.labels_v3 = labels_v3
            self.labeller = Labeller(self.y_emb.max_bow_genre_size, self.n_tokens, self.sample_length, v3=self.labels_v3)
        else:
            self.labeller = EmptyLabeller()

        print(f"Level:{level}, Cond downsample:{self.cond_downsample}, Raw to tokens:{self.raw_to_tokens}, Sample length:{self.sample_length}")


    def get_y(self, labels, start, get_indices=False):
        if isinstance(self.labeller, EmptyLabeller):
            return None
        y = labels['y'].clone()

        # Set sample_length to match this level
        y[:, 2] = int(self.sample_length)

        # Set offset
        y[:, 1:2] = y[:, 1:2] + int(start * self.raw_to_tokens)

        # Set lyric tokens
        indices = self.labeller.set_y_lyric_tokens(y, labels)
        if get_indices:
            return y, indices
        else:
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

    def prior_preprocess(self, xs, conds):
        N = xs[0].shape[0]
        for i in range(len(xs)):
            x, shape, dims = xs[i], self.prior_shapes[i], self.prior_dims[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            assert isinstance(x, t.cuda.LongTensor), x
            assert (0 <= x).all() and (x < bins).all()
            #assert_shape(x, (N, *shape))
            xs[i] = (xs[i] + bins_shift).view(N, -1)

        for i in range(len(conds)):
            cond, shape, dims = conds[i], self.prior_shapes[i], self.prior_dims[i]
            if cond is not None:
                assert_shape(cond, (N, dims, self.prior_width))
            else:
                conds[i] = t.zeros((N, dims, self.prior_width), dtype=t.float, device='cuda')

        return t.cat(xs, dim=1), t.cat(conds, dim=1)

    def prior_postprocess(self, z):
        N = z.shape[0]
        dims = (self.prior_dims[0], z.shape[1] - self.prior_dims[0])
        # xs = list(t.split(z, self.prior_dims, dim=1))
        xs = list(t.split(z, dims, dim=1))

        for i in range(len(xs)):
            # x, shape, dims, bins, bins_shift = xs[i], self.prior_shapes[i], self.prior_dims[i], self.prior_bins[i], self.prior_bins_shift[i]
            # assert_shape(x, (N, dims))
            shape = self.prior_shapes[i]
            bins, bins_shift = int(self.prior_bins[i]), int(self.prior_bins_shift[i])
            # xs[i] = (xs[i] - bins_shift).view(N, *shape) #view(N, -1, *shape[1:])
            xs[i] = (xs[i] - bins_shift).view(N, -1, *shape[1:])
            xs[i] = t.clamp(xs[i], min=0)  # If not masking loss, model may have generated lyric/midi tokens which are now shifted <0 by bin_shift
            assert (xs[i] < bins).all(), f'rank: {dist.get_rank()}, bins: {bins}, dims {dims}, shape {shape}, prior_shape {self.prior_shapes}, bins_shift {bins_shift}, xs[i]: {xs[i]}'

        return xs[-1]

    def x_emb(self, z_conds):
        z_conds = z_conds[:self.cond_level - self.level]
        assert len(z_conds) == len(self.conditioner_blocks) == self.cond_level - self.level, f"Expected {len(z_conds)} == {len(self.conditioner_blocks)} == {self.cond_level} - {self.level}"
        x_cond = None
        for z_cond, conditioner_block in reversed(list(zip(z_conds, self.conditioner_blocks))):
            x_cond = conditioner_block(z_cond, x_cond)
        return x_cond

    def encode(self, x, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels
        # Get latents
        with t.no_grad():
            zs = self.encoder(x, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return zs

    def decode(self, zs, start_level=None, end_level=None, bs_chunks=1):
        if start_level == None:
            start_level = self.level
        if end_level == None:
            end_level = self.levels

        assert len(zs) == end_level - start_level
        with t.no_grad():
            x_out = self.decoder(zs, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return x_out

    def get_cond(self, z_conds, y):
        if y is not None:
            assert y.shape[1] == 4 + self.y_emb.max_bow_genre_size + self.n_tokens, f"Expected {4} + {self.y_emb.max_bow_genre_size} + {self.n_tokens}, got {y.shape[1]}"
            n_labels = y.shape[1] - self.n_tokens
            y, prime = y[:,:n_labels], y[:,n_labels:]
        else:
            y, prime = None, None
        y_cond, y_pos = self.y_emb(y) if self.y_cond else (None, None)
        x_cond = self.x_emb(z_conds) if self.x_cond else y_pos
        return x_cond, y_cond, prime

    def sample(self, n_samples, z=None, z_conds=None, y=None, fp16=False, temp=1.0, top_k=0, top_p=0.0,
               chunk_size=None, sample_tokens=None):
        N = n_samples
        if z is not None: assert z.shape[0] == N, f"Expected shape ({N},**), got shape {z.shape}"
        if y is not None: assert y.shape[0] == N, f"Expected shape ({N},**), got shape {y.shape}"
        if z_conds is not None:
            for z_cond in z_conds:
                assert z_cond.shape[0] == N,  f"Expected shape ({N},**), got shape {z_cond.shape}"

        no_past_context = (z is None or z.shape[1] == 0)
        if dist.get_rank() == 0:
            name = {True: 'Ancestral', False: 'Primed'}[no_past_context]
            print(f"{name} sampling {n_samples} samples with temp={temp}, top_k={top_k}, top_p={top_p}")

        with t.no_grad():
            # Currently x_cond only uses immediately above layer
            x_cond, y_cond, prime = self.get_cond(z_conds, y)
            if self.single_enc_dec:
                # assert chunk_size % self.prime_loss_dims == 0. TODO: Check if needed
                if no_past_context:
                    z, x_cond = self.prior_preprocess([prime], [None, x_cond])
                else:
                    z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
                if sample_tokens is not None:
                    sample_tokens += self.n_tokens
                z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, fp16=fp16, temp=temp,
                                             top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
                z = self.prior_postprocess(z)
            else:
                encoder_kv = self.get_encoder_kv(prime, fp16=fp16, sample=True)
                if no_past_context:
                    z = self.prior.sample(n_samples, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp, top_k=top_k,
                                          top_p=top_p, sample_tokens=sample_tokens)
                else:
                    z = self.prior.primed_sample(n_samples, z, x_cond, y_cond, encoder_kv, fp16=fp16, temp=temp,
                                             top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
            if sample_tokens is None:
                assert_shape(z, (N, *self.z_shape))
        return z

    def get_encoder_kv(self, prime, fp16=False, sample=False):
        if self.n_tokens != 0 and self.use_tokens:
            if sample:
                self.prime_prior.cuda()
            N = prime.shape[0]
            prime_acts = self.prime_prior(prime, None, None, None, fp16=fp16)
            assert_shape(prime_acts, (N, self.prime_loss_dims, self.prime_acts_width))
            assert prime_acts.dtype == t.float, f'Expected t.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))
            assert encoder_kv.dtype == t.float, f'Expected t.float, got {encoder_kv.dtype}'
            if sample:
                self.prime_prior.cpu()
                if fp16:
                    encoder_kv = encoder_kv.half()
        else:
            encoder_kv = None
        return encoder_kv

    def get_prime_loss(self, encoder_kv, prime_t):
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(encoder_kv.view(-1, self.prime_bins), prime_t.view(-1)) / np.log(2.)
        else:
            prime_loss = t.tensor(0.0, device='cuda')
        return prime_loss

    def z_forward(self, z, z_conds=[], y=None, fp16=False, get_preds=False, get_attn_weights=False):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        x_cond, y_cond, prime = self.get_cond(z_conds, y)
        if self.copy_input:
            prime = z[:,:self.n_tokens]
        if self.single_enc_dec:
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
            (prime_loss, gen_loss), preds = self.prior(z, x_cond, y_cond, fp16=fp16, get_sep_loss=True, get_preds=get_preds)
        else:
            encoder_kv = self.get_encoder_kv(prime, fp16=fp16)
            prime_loss = self.get_prime_loss(encoder_kv, prime)
            gen_loss, preds = self.prior(z, x_cond, y_cond, encoder_kv, fp16=fp16, get_preds=get_preds)
        loss = (self.prime_loss_fraction*prime_loss*self.prime_loss_dims/self.total_loss_dims) + \
                   (gen_loss*self.gen_loss_dims/self.total_loss_dims)
        metrics=dict(bpd=gen_loss.clone().detach(), prime_loss=prime_loss.clone().detach(),
                     gen_loss=gen_loss.clone().detach())
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            return loss, metrics

    def forward(self, x, y=None, fp16=False, decode=False, get_preds=False):
        bs = x.shape[0]
        z, *z_conds = self.encode(x, bs_chunks=bs)
        loss, metrics = self.z_forward(z=z, z_conds=z_conds, y=y, fp16=fp16, get_preds=get_preds)
        if decode:
            x_out = self.decode([z, *z_conds])
        else:
            x_out = None
        return x_out, loss, metrics
