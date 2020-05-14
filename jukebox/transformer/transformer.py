import functools
import numpy as np
import torch as t
import torch.nn as nn
import jukebox.utils.dist_adapter as dist

from jukebox.transformer.ops import Conv1D, ACT_FNS, LayerNorm
from jukebox.transformer.factored_attention import FactoredAttention
from jukebox.utils.checkpoint import checkpoint

def _convert_mlp_traced(l):
    if isinstance(l, ResAttnBlock):
        l.mlp = t.jit.trace(l.mlp, t.randn(1, 1, l.n_in).cuda())

def _convert_mlp_traced_fp16(l):
    if isinstance(l, ResAttnBlock):
        l.mlp = t.jit.trace(l.mlp, t.randn(1, 1, l.n_in).cuda().half())

class MLP(nn.Module):
    def __init__(self, n_in, n_state, resid_dropout=0.0, afn='quick_gelu', zero_out=False, init_scale=1.0):
        super().__init__()
        self.c_fc = Conv1D(n_in, n_state, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.act = ACT_FNS[afn]
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x

    def forward(self, x):
        m = self.act(self.c_fc(x))
        m = self.c_proj(m)
        return self.resid_dropout(m)

class ResAttnBlock(nn.Module):
    def __init__(self, n_in, n_ctx, n_head,
                 attn_dropout=0.0, resid_dropout=0.0,
                 afn='quick_gelu', scale=True, mask=False,
                 zero_out=False, init_scale=1.0, res_scale=1.0,
                 m_attn = 0.25, m_mlp = 1.,
                 checkpoint_attn = 0, checkpoint_mlp = 0,
                 attn_func=0, blocks=None, spread=None,
                 encoder_dims=None, prime_len=None):
        super().__init__()
        self.attn = FactoredAttention(n_in=n_in, n_ctx=n_ctx, n_state=int(m_attn * n_in), n_head=n_head,
                                      attn_dropout=attn_dropout, resid_dropout=resid_dropout,
                                      scale=scale, mask=mask,
                                      zero_out=zero_out, init_scale=init_scale,
                                      checkpoint_attn=checkpoint_attn,
                                      attn_func=attn_func, blocks=blocks, spread=spread,
                                      encoder_dims=encoder_dims, prime_len=prime_len)
        self.ln_0 = LayerNorm(n_in)
        self.mlp = MLP(n_in=n_in, n_state=int(m_mlp * n_in),
                       resid_dropout=resid_dropout,
                       afn=afn,
                       zero_out=zero_out, init_scale=init_scale)
        self.ln_1 = LayerNorm(n_in)
        self.res_scale = res_scale

        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_mlp = checkpoint_mlp
        self.n_in = n_in
        self.attn_func = attn_func

    def forward(self, x, encoder_kv, sample=False):
        if sample:
            a = self.attn(self.ln_0(x), encoder_kv, sample)
            m = self.mlp(self.ln_1(x + a))
        else:
            if self.attn_func == 6:
                assert encoder_kv is not None
                a = checkpoint(lambda _x,_enc_kv,_s=sample: self.attn(self.ln_0(_x),_enc_kv,_s),
                               (x,encoder_kv),
                               (*self.attn.parameters(), *self.ln_0.parameters()),
                               self.checkpoint_attn == 3)  # 2 recomputes after the projections, and 1 recomputes after head splitting.
            else:
                assert encoder_kv is None
                a = checkpoint(lambda _x,_enc_kv=None,_s=sample: self.attn(self.ln_0(_x),_enc_kv,_s),
                               (x,),
                               (*self.attn.parameters(), *self.ln_0.parameters()),
                               self.checkpoint_attn == 3)  # 2 recomputes after the projections, and 1 recomputes after head splitting.
            m = checkpoint(lambda _x: self.mlp(self.ln_1(_x)), (x + a,),
                           (*self.mlp.parameters(), *self.ln_1.parameters()),
                           self.checkpoint_mlp == 1)
        if self.res_scale == 1.0:
            h = x + a + m
        else:
            h = x + self.res_scale * (a + m)
        return h

class Transformer(nn.Module):
    def __init__(self, n_in, n_ctx, n_head, n_depth,
                 attn_dropout=0.0, resid_dropout=0.0,
                 afn='quick_gelu', scale=True, mask=False,
                 zero_out=False, init_scale=1.0, res_scale=False,
                 m_attn=0.25, m_mlp=1.,
                 checkpoint_attn=0, checkpoint_mlp=0, checkpoint_res=0,
                 attn_order=0, blocks=None, spread=None,
                 encoder_dims=None, prime_len=None):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.encoder_dims = encoder_dims
        self.blocks = blocks
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.prime_len = prime_len
        self.n_head = n_head

        res_scale = 1.0 / n_depth if res_scale else 1.0

        # Orders of attn_func
        attn_func = {0: lambda d: 0,                    # Complete dense attn
                     1: lambda d: [1,2][d%2],           # Alternate row and column attn
                     2: lambda d: [1,2,3][d % 3],       # Alternate row, column and previous row attn
                     3: lambda d: [1,4][d % 2],         # Alternate row and last column
                     4: lambda d: [1,5][d % 2],         # Alternate row and last k columns
                     5: lambda d: [1,4,1,1][d % 4],      # Alternate row, last column, row, row
                     6: lambda d: [1,2,3,6][d % 4],
                     7: lambda d: [*[1,2,3]*5,6][d%16],
                     8: lambda d: [1,2,3,1,2,3,1,2,3,6][d%10], # Used by separated_enc_dec model with lyrics
                     9: lambda d: [1,2,3,0][d % 4],
                     10: lambda d: [*[1,2,3,1,2,3,1,2,3],*[1,2,3,1,2,3,1,2,3,6]*7][d%79], # Used by large separated_enc_dec model with lyrics
                     11: lambda d: [6,6,0][d%3] if d%16 == 15 else [1,2,3][d%3],
                     12: lambda d: [7,7,0][d%3] if d%16 == 15 else [1,2,3][d%3], # Used by single_enc_dec model with lyrics
                     }[attn_order]

        attn_cycle = {0:1, 1:2, 2:3, 3:2, 4:2, 5:4, 6:4, 7:16, 8:10, 9:4, 10:79, 11:16, 12:16}[attn_order]
        #assert n_depth % attn_cycle == 0, f'Depth {n_depth} not a multiple of cycle {attn_cycle} for attn_order {attn_order}'

        attn_block = lambda d: ResAttnBlock(n_in=n_in, n_ctx=n_ctx, n_head=n_head,
                                  attn_dropout=attn_dropout, resid_dropout=resid_dropout,
                                  afn=afn, scale=scale, mask=mask,
                                  zero_out=zero_out if attn_func(d) !=6 else True,
                                  init_scale=init_scale, res_scale=res_scale,
                                  m_attn=m_attn, m_mlp=m_mlp,
                                  checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp,
                                  attn_func=attn_func(d), blocks=blocks, spread=spread,
                                  encoder_dims=encoder_dims, prime_len=prime_len)

        self.checkpoint_res = checkpoint_res
        self._attn_mods = nn.ModuleList()
        for d in range(n_depth):
            self._attn_mods.append(attn_block(d))
        self.ws = []


    def set_record_attn(self, record_attn):
        """
        Arguments:
            record_attn (bool or set): Makes forward prop dump self-attention
                softmaxes to self.ws. Either a set of layer indices indicating
                which layers to store, or a boolean value indicating whether to
                dump all.
        """
        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn
        for i, l in enumerate(self._attn_mods):
            l.attn.record_attn = _should_record_attn(i)
        if record_attn:
            assert self.ws == []
            for l in self._attn_mods:
                assert l.attn.w == None
        else:
            self.ws = []
            for l in self._attn_mods:
                l.attn.w = None

    def forward(self, x, encoder_kv=None, sample=False, fp16=False, fp16_out=False):
        if fp16:
            x = x.half()

        # Blocks
        for i,l in enumerate(self._attn_mods):
            if self.checkpoint_res == 1 and not sample:
                if l.attn_func == 6:
                    assert encoder_kv is not None
                    f = functools.partial(l, sample=sample)
                    x = checkpoint(f, (x, encoder_kv), l.parameters(), True)
                else:
                    f = functools.partial(l, encoder_kv=None, sample=sample)
                    x = checkpoint(f, (x,), l.parameters(), True)
            else:
                if l.attn_func == 6:
                    x = l(x, encoder_kv=encoder_kv, sample=sample)
                else:
                    x = l(x, encoder_kv=None, sample=sample)
            if l.attn.record_attn:
                self.ws.append(l.attn.w)
        if not fp16_out:
            x = x.float()
        return x

    def check_cache(self, n_samples, sample_t, fp16):
        for l in self._attn_mods:
            l.attn.check_cache(n_samples, sample_t, fp16)

    def del_cache(self):
        for l in self._attn_mods:
            l.attn.del_cache()

    def check_sample(self):
        bs, l, s, d = (4, self.n_ctx, self.encoder_dims, self.n_in)
        prime = 5
        with t.no_grad():
            encoder_kv = t.randn(bs, s, d).cuda()
            x = t.randn(bs, l, d).cuda()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=True)

            self.del_cache()
            x_chunks = t.chunk(x, 4, dim=1)
            y_chunks = []
            n = 0
            for x_chunk in x_chunks:
                self.check_cache(bs, n, False)
                y_chunk = self.forward(x_chunk, encoder_kv=encoder_kv, sample=True)
                y_chunks.append(y_chunk)
                n += x_chunk.shape[1]
            self.check_cache(bs, n, False)
            y_forw_in_chunks = t.cat(y_chunks, dim=1)

            max_err = t.max(t.abs(y_forw - y_forw_in_chunks))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-6]}"


if __name__ == '__main__':
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    setup_dist_from_mpi(port=29600)
    n_in = 16
    n_ctx = 192
    n_head = 4
    n_depth = 12
    blocks = 16
    for attn_order in [0,2,6]:
        encoder_dims = {0: 0, 2: 0, 6: 64}[attn_order]
        prior = Transformer(n_in, n_ctx, n_head, n_depth, mask=True, attn_order=attn_order, encoder_dims=encoder_dims, blocks=blocks).cuda()
        prior.training = False
        prior.check_sample()
        print(f"Checked attn_order: {attn_order}")
