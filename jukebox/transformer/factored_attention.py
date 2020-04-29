# Factored attention
import math
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from jukebox.transformer.ops import Conv1D
from jukebox.utils.checkpoint import checkpoint

def repeat(x, n, dim):
    if dim == -1:
        dim = len(x.shape) - 1
    return x.view(int(np.prod(x.shape[:dim+1])), 1, int(np.prod(x.shape[dim+1:]))).repeat(1,n,1).view(*x.shape[:dim], n * x.shape[dim], *x.shape[dim+1:])

def get_mask(mask, q_l, kv_l, blocks, spread, device, sample, sample_t):
    # returns a mask of shape 1 x 1 x q_l x kv_l or None if masking is not needed.
    if mask is None or q_l == 1:
        return None
    offset = sample_t - q_l if sample else max(kv_l - q_l, 0)
    if mask == 'autoregressive':
        # Masked dense
        mask = t.ones(q_l, kv_l, device=device).tril(offset)
    elif mask == 'summary':
        # Masked summary
        mask = t.nn.functional.pad(t.ones(q_l, q_l, device=device).tril().view(q_l, blocks, q_l // blocks)[:,:-1,-kv_l//blocks:],(0,0,1,0),value=1).contiguous().view(q_l, kv_l)
    elif mask == 'prime':
        mask = t.ones(q_l, kv_l, device=device).tril(offset)
    return mask.view(1,1,q_l,kv_l)

class FactoredAttention(nn.Module):
    def __init__(self, n_in, n_ctx, n_state, n_head,
                 attn_dropout=0.0, resid_dropout=0.0,
                 scale=True, mask=False,
                 zero_out=False, init_scale=1.0,
                 checkpoint_attn=0,
                 attn_func=0, blocks=None, spread=None,
                 encoder_dims=None, prime_len=None):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx # NOTE: n_ctx could be different within operations. This is complete n_ctx
        self.n_state = n_state
        assert n_state % n_head == 0
        self.n_head = n_head
        self.scale = scale
        self.mask = mask
        if attn_func == 6:
            self.c_attn = Conv1D(n_in, n_state, init_scale=init_scale)
            self.c_enc_kv = Conv1D(n_in, n_state * 2, init_scale=init_scale)
        else:
            self.c_attn = Conv1D(n_in, n_state * 3, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else lambda x: x
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x

        # Sequence of length l is factored as [blocks, l // blocks]
        self.attn_func = attn_func
        self.qkv, self.attn, self.attn_mask = {
            0: (self.factored_qkv, self.dense_attn, 'autoregressive'),              # Attend to all positions
            1: (self.factored_qkv, self.block_attn, 'autoregressive'),              # Attend to your block
            2: (self.factored_qkv, self.transpose_block_attn, 'autoregressive'),    # Attend to transpose block
            3: (self.factored_qkv, self.prev_block_attn, None),                     # Attend to previous block
            4: (self.factored_qkv, self.summary_attn, 'summary'),                   # Attend to last position of each block
            5: (self.factored_qkv, self.summary_spread_attn, 'summary'),
            6: (self.decode_qkv, self.decode_attn, None),
            7: (self.prime_qkv, self.prime_attn, 'prime')
        }[attn_func] # Attend to last k position of each block

        self.blocks = blocks
        self.spread = spread
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.checkpoint_attn = checkpoint_attn # 0: None, 1: Attn after heads split, 2: Attn

        self.sample_t = 0
        self.cache = {}
        self.encoder_dims = encoder_dims
        self.prime_len = prime_len
        self.record_attn = False
        self.w = None

    def _attn(self, q, k, v, sample):
        scale = 1. / math.sqrt(math.sqrt(self.n_state // self.n_head))
        if self.training:
            w = t.matmul(q * scale, k * scale)
        else:
            w = t.matmul(q, k)
            w.mul_(scale*scale)
        wtype = w.dtype
        w = w.float()
        if self.mask:
            # Generate appropriate mask to mask out all positions before current
            # Might take up lot of memory for dense, so can cache it
            mask = get_mask(self.attn_mask, q.size(-2), k.size(-1), self.blocks, self.spread, w.device, sample, self.sample_t)
            if mask is not None:
                #print(mask)
                w = w * mask + -1e9 * (1 - mask)
            w = F.softmax(w, dim=-1).type(wtype)
        else:
            w = F.softmax(w, dim=-1).type(wtype)
        if self.record_attn:
            self.w = w #.float().cpu().numpy()
            if self.attn_func == 7:
                # only keep music queries and lyrics keys/values
                self.w = self.w[:,:,self.prime_len:,:self.prime_len]
        w = self.attn_dropout(w)
        a = t.matmul(w, v)
        return a

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = (*x.size()[:-2], x.size(-2) * x.size(-1))
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = (*x.size()[:-1], self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def dense_attn(self, query, key, value, sample):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if self.checkpoint_attn == 1 and not sample:
            a = checkpoint(lambda q,k,v,s=sample: self._attn(q,k,v,s), (query, key, value),
                       (), True)
        else:
            a = self._attn(query,key,value,sample)
        a = self.merge_heads(a)
        return a

    def block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert l == self._suff_cache_len(), f"{l} != {self._suff_cache_len()}"
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            if ql < l:
                l = ql
                k = k[:, -l:].contiguous()
                v = v[:, -l:].contiguous()
            k = k.view(bs * l // block_ctx, block_ctx, d)
            v = v.view(bs * l // block_ctx, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def transpose_block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            block_l = (l - 1) % block_ctx
            k = k[:,block_l::block_ctx,:]
            v = v[:,block_l::block_ctx,:]
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs, ql // block_ctx, block_ctx, d).transpose(1,2).contiguous().view(bs * block_ctx, ql // block_ctx, d)
            k = k.view(bs,  l // block_ctx, block_ctx, d).transpose(1,2).contiguous().view(bs * block_ctx,  l // block_ctx, d)
            v = v.view(bs,  l // block_ctx, block_ctx, d).transpose(1,2).contiguous().view(bs * block_ctx,  l // block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, block_ctx, ql // block_ctx, d).transpose(1,2).contiguous().view(bs, ql, d)

    def prev_block_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert l == self._suff_cache_len(), f"{l} != {self._suff_cache_len()}"
            block = (l - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            if block > 0:
                assert prev_l == 0
                k = k[:, prev_l:prev_l + block_ctx, :]
                v = v[:, prev_l:prev_l + block_ctx, :]
            else:
                k = t.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
                v = t.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            k = t.nn.functional.pad(k.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0,0,0,0,1,0)).view(bs * l // block_ctx, block_ctx, d)
            v = t.nn.functional.pad(v.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0,0,0,0,1,0)).view(bs * l // block_ctx, block_ctx, d)
            if ql < l:
                qb = ql // block_ctx
                kb =  l // block_ctx
                l = ql
                k = k.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
                v = v.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_attn(self, q, k, v, sample):
        blocks, block_ctx = self.blocks, self.block_ctx # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            k = t.nn.functional.pad(k[:, block_ctx-1:blocks*block_ctx-1:block_ctx, :],(0,0,1,0))
            v = t.nn.functional.pad(v[:, block_ctx-1:blocks*block_ctx-1:block_ctx, :],(0,0,1,0))
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = t.nn.functional.pad(k.view(bs, blocks, l // blocks, d)[:, :-1, -1, :],(0,0,1,0)) # bs, blocks, d
            v = t.nn.functional.pad(v.view(bs, blocks, l // blocks, d)[:, :-1, -1, :],(0,0,1,0)) # bs, blocks, d
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_spread_attn(self, q, k, v, sample):
        blocks, block_ctx, spread = self.blocks, self.block_ctx, self.spread # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert False, "Not yet implemented"
            # k = t.nn.functional.pad(k,(0,0,block_ctx,(-l)%block_ctx)).view(bs, -1, block_ctx, d)[:,:-1,-spread:,:].contiguous().view(bs, -1, d)
            # v = t.nn.functional.pad(v,(0,0,block_ctx,(-l)%block_ctx)).view(bs, -1, block_ctx, d)[:,:-1,-spread:,:].contiguous().view(bs, -1, d)
            # return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = t.nn.functional.pad(k.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :],(0,0,0,0,1,0)).contiguous().view(bs, blocks * spread, d)  # bs, blocks * spread, d
            v = t.nn.functional.pad(v.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :],(0,0,0,0,1,0)).contiguous().view(bs, blocks * spread, d)  # bs, blocks * spread, d
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def prime_attn(self, q, k, v, sample):
        prime_len = self._prime_len
        k = k[:, :prime_len]
        v = v[:, :prime_len]
        return self.dense_attn(q, k, v, sample)

    def decode_attn(self, q, k, v, sample):
        assert k.shape[1] == v.shape[1] == self.encoder_dims, f'k: {k.shape}, v: {v.shape}, enc_dims: {self.encoder_dims}'
        return self.dense_attn(q, k, v, sample)

    def factored_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            self.sample_t += curr_ctx
            key, value = self._append_cache(key, value)
            l_cache = self._suff_cache_len()
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            if curr_ctx > 1:
                if self.attn_func != 0:
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                    assert key.shape[1] % self.block_ctx == 0
                    assert query.shape[1] % self.block_ctx == 0
                assert key.shape[1] == value.shape[1]
                assert query.shape[1] <= key.shape[1]
                sample = False
            else:
                key = self.cache['key']
                value = self.cache['value']
        return query, key, value, sample

    def prime_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            if self._cache_len() < self._prime_len:
                self._append_cache(key, value)
            if self._cache_len() > self._prime_len:
                self._slice_cache(0, self._prime_len)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
            assert key.shape[1] == value.shape[1] == self._suff_cache_len(), f'k: {key.shape}, v: {value.shape}, prime_dims: {self._suff_cache_len()}'
        else:
            assert key.shape[1] == value.shape[1] == self.n_ctx, f'k: {key.shape}, v: {value.shape}, prime_dims: {self.n_ctx}'
        assert key.shape[0] == value.shape[0] == query.shape[0], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert key.shape[2] == value.shape[2] == query.shape[2], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def decode_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is not None
        query = x
        if sample:
            if self.sample_t == 0:
                self.cache['key'], self.cache['value'] = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
        else:
            key, value = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
        assert key.shape[0] == value.shape[0] == query.shape[0], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert key.shape[1] == value.shape[1] == self.encoder_dims, f'k: {key.shape}, v: {value.shape}, enc_dims: {self.encoder_dims}'
        assert key.shape[2] == value.shape[2] == query.shape[2], f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def forward(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        x = self.c_attn(x)
        query, key, value, sample = self.qkv(x, encoder_kv=encoder_kv, sample=sample)
        if self.checkpoint_attn == 2 and not sample:
            a = checkpoint(lambda q,k,v,s=sample: self.attn(q,k,v,s), (query, key, value), (), True)
        else:
            a = self.attn(query,key,value,sample)
        if a.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            a = a[:,offset:offset + curr_ctx,:].contiguous()
        a = self.c_proj(a)
        return self.resid_dropout(a)

    @property
    def _prime_len(self):
        prime_len = self.prime_len
        assert prime_len is not None
        prime_blocks = (prime_len // self.blocks) + 1
        return prime_blocks * self.blocks

    def _offset(self, curr_ctx):
        if self.attn_func == 0:
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    def _pad_to_block_ctx(self, x, query=False):
        l = x.shape[1]
        offset = self._offset(l) if query else 0
        n_blocks = (l + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - l - offset
        if pad == 0 and offset == 0:
            return x
        else:
            return F.pad(x, (0, 0, offset, pad))

    def _cache_len(self):
        return 0 if 'key' not in self.cache else self.cache['key'].shape[1]

    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and
            self.sample_t reflects the 1-indexed sample location in the
            context.
        """
        if self.attn_func == 0:
            return self.sample_t
        elif self.attn_func == 1:
            return (self.sample_t - 1) % self.block_ctx + 1
        elif self.attn_func == 2:
            return self.sample_t
        elif self.attn_func == 3:
            if self.sample_t <= self.block_ctx:
                return self.sample_t
            else:
                curr_block = (self.sample_t - 1) % self.block_ctx + 1
                prev_block = self.block_ctx
                return curr_block + prev_block
        elif self.attn_func == 6:
            return self.encoder_dims
        elif self.attn_func == 7:
            return min(self.sample_t, self._prime_len)
        else:
            raise NotImplementedError()

    def _slice_cache(self, start, end=None):
        self.cache['key'] = self.cache['key'][:, start:end]
        self.cache['value'] = self.cache['value'][:, start:end]

    def _append_cache(self, key, value):
        if 'key' not in self.cache:
            self.cache['key'] = key
            self.cache['value'] = value
        else:
            old_key, old_value = key, value
            key = t.cat([self.cache['key'], key], dim=1)
            value = t.cat([self.cache['value'], value], dim=1)
            del self.cache['key']
            del self.cache['value']
            del old_key
            del old_value
            self.cache['key'] = key
            self.cache['value'] = value
        return self.cache['key'], self.cache['value']

    def del_cache(self):
        self.sample_t = 0
        if 'key' in self.cache:
            del self.cache['key']
        if 'value' in self.cache:
            del self.cache['value']
        self.cache = {}

    def check(self):
        blocks = self.blocks or 1
        spread = self.spread or 1
        bs, l, d = (4, self.n_ctx, self.n_in)
        x = t.randn(bs, l, d).cuda()
        x.requires_grad = True
        x_out = self.forward(x) # bs, l, d
        loss = x_out.mean(dim = -1) # bs, l
        pos = 60
        grad = t.autograd.grad(loss[2, pos], x)[0]

        assert grad.shape == (bs, l, d)
        assert (grad[:2] == 0).all()
        assert (grad[3:] == 0).all()
        assert (grad[2, (pos + 1):] == 0).all()
        pos_grad = (t.sum(grad[2] ** 2, dim=-1) > 0).nonzero().view(-1).cpu()

        block_pos = pos - (pos % (l // blocks))
        exp_pos_grad = {0: t.arange(pos),
                        1: t.arange(block_pos, pos),
                        2: t.arange(pos % (l // blocks), pos, l // blocks),
                        3: t.arange(block_pos - l // blocks, block_pos),
                        4: t.arange(l // blocks - 1, pos, l // blocks),
                        5: ((t.arange(pos) % (l // blocks) >= (l // blocks - spread)) & (t.arange(pos) < block_pos)).nonzero().view(-1)}[self.attn_func]
        exp_pos_grad = t.cat([exp_pos_grad, t.tensor([pos])], dim=-1)

        assert (len(pos_grad) == len(exp_pos_grad)) and (pos_grad == exp_pos_grad).all(), \
            f"Expected pos grad {exp_pos_grad} got {pos_grad} for attn_func {self.attn_func} pos {pos} l {l} blocks {blocks}"

    def check_cache(self, n_samples, sample_t, fp16):
        assert self.sample_t == sample_t, f"{self.sample_t} != {sample_t}"
        if sample_t == 0:
            assert self.cache == {}
        else:
            dtype = {True: t.float16, False: t.float32}[fp16]
            l_cache = self._suff_cache_len()
            assert self.cache['key'].shape == (n_samples, l_cache, self.n_state)
            assert self.cache['value'].shape == (n_samples, l_cache, self.n_state)
            assert self.cache['key'].dtype == dtype, f"Expected {dtype}, got {self.cache['key'].dtype}"
            assert self.cache['value'].dtype == dtype, f"Expected {dtype}, got {self.cache['value'].dtype}"

    def check_sample(self):
        t.manual_seed(42)
        bs, l, d = (4, self.n_ctx, self.n_in)
        prime = 5
        x = t.randn(bs, l, d).cuda()
        xs = t.chunk(x, l, dim=1)
        assert self.sample_t == 0
        assert self.cache == {}

        with t.no_grad():
            enc_l = self.encoder_dims
            encoder_kv = None
            if self.attn_func == 6:
                encoder_kv = t.randn(bs, enc_l, d).cuda()

            # Normal path
            x_out_normal = self.forward(x, encoder_kv=encoder_kv)

            # Sampling path
            x_out_sample = t.cat([self.forward(xs[i], encoder_kv=encoder_kv, sample=True) for i in range(l)],dim=1)
        max_err = t.max(t.abs(x_out_sample - x_out_normal))
        assert max_err < 1e-8, f"Max sampling err is {max_err} {[i for i in range(l) if t.max(t.abs(x_out_sample - x_out_normal)[:,i,:]) > 1e-8]}"

        with t.no_grad():
            x_out_normal = x_out_normal[:,:prime,:]
            # Prime sampling path
            self.del_cache()
            x_out_sample = self.forward(x[:,:prime,:].contiguous(), encoder_kv=encoder_kv, sample=True)
            self.check_cache(bs, prime, False)

        max_err = t.max(t.abs(x_out_sample - x_out_normal))
        assert max_err < 1e-8, f"Max prime sampling err is {max_err} {[i for i in range(prime) if t.max(t.abs(x_out_sample - x_out_normal)[:,i,:]) > 1e-8]}"

    def check_chunks(self, chunk_size):
        t.manual_seed(42)
        bs, l, d = (4, self.n_ctx, self.n_in)
        enc_l = self.encoder_dims
        assert l % chunk_size == 0
        n_chunks = l // chunk_size
        with t.no_grad():
            encoder_kv = None
            x = t.randn(bs, l, d).cuda()
            if self.attn_func == 6:
                encoder_kv = t.randn(bs, enc_l, d).cuda()

            self.del_cache()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=False)
            self.del_cache()
            y_forw_sample = self.forward(x, encoder_kv=encoder_kv, sample=True)
            max_err = t.max(t.abs(y_forw - y_forw_sample))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_sample)[:, i, :]) > 1e-6]}"

            self.del_cache()
            x_chunks = t.chunk(x, n_chunks, dim=1)
            y_chunks = []
            total_len = 0
            for x_chunk in x_chunks:
                y_chunk = self.forward(x_chunk.contiguous(), encoder_kv=encoder_kv, sample=True)
                total_len += x_chunk.shape[1]
                self.check_cache(bs, total_len, False)
                y_chunks.append(y_chunk)
            y_forw_in_chunks = t.cat(y_chunks, dim=1)

            max_err = t.max(t.abs(y_forw - y_forw_in_chunks))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-6]}"


if __name__ == '__main__':
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    setup_dist_from_mpi(port=29600)
    n_in = 16
    n_state = n_in * 2
    n_ctx = 6144
    n_head = 4
    n_depth = 12
    blocks = 64
    chunk_size = 8
    for attn_func in [0, 1, 2, 3, 6, 7]:
        encoder_dims = {0: 0, 1: 0, 2: 0, 3: 0, 6: 64, 7: 0}[attn_func]
        prime_len = {0: 0, 1: 0, 2: 0, 3: 0, 6: 0, 7: 384}[attn_func]
        attn = FactoredAttention(n_in, n_ctx + prime_len, n_state, n_head, mask=True,
                                 attn_func=attn_func, blocks=blocks,
                                 encoder_dims=encoder_dims, prime_len=prime_len)
        attn.training = False
        attn.check_sample()
        attn.check_chunks(chunk_size)
        print(f"Checked attn_func: {attn_func}")
