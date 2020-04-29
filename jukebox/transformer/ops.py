import math
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

# Import FusedLayerNorm if we have apex, otherwise use regular LayerNorm
try:
    from apex.normalization import FusedLayerNorm
    print("Using apex FusedLayerNorm")
except ImportError:
    from torch.nn import LayerNorm as FusedLayerNorm

class LayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.width = np.prod(normalized_shape)
        self.max_numel = 65535*self.width

    def forward(self, input):
        if input.numel() > self.max_numel:
            return F.layer_norm(input.float(), self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)
        else:
            return super(LayerNorm, self).forward(input.float()).type_as(input)

def gelu(x):
    return 0.5 * x * (1 + t.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * t.pow(x, 3))))


def swish(x):
    return x * t.sigmoid(x)

@t.jit.script
def quick_gelu(x):
    return x * t.sigmoid(1.702 * x)

@t.jit.script
def quick_gelu_bwd(x, grad_output):
    sig = t.sigmoid(1.702 * x)
    return grad_output * sig * (1.702 * x * (1 - sig) + 1.)

class QuickGelu(t.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return quick_gelu(x)

    @staticmethod
    def backward(ctx, grad_output):
        return quick_gelu_bwd(ctx.saved_tensors[0], grad_output)

def memory_efficient_quick_gelu(x):
    return QuickGelu.apply(x)

ACT_FNS = {
    'relu': t.nn.functional.relu,
    'swish': swish,
    'gelu': gelu,
    'quick_gelu': memory_efficient_quick_gelu #quick_gelu
}

def _move_to_gpu_and_convert_conv_weights_to_fp16(l):
    l.cuda()
    if isinstance(l, Conv1D):
        l.w.data = l.w.data.half()

def _convert_conv_weights_to_fp32(l):
    if isinstance(l, Conv1D):
        l.w.data = l.w.data.float()

def _convert_conv_weights_to_fp16(l):
    if isinstance(l, Conv1D):
        l.w.data = l.w.data.half()

def _convert_embedding_weights_to_fp16(l):
    if isinstance(l, t.nn.Embedding):
        l.weight.data = l.weight.data.half()

def _convert_embedding_weights_to_fp32(l):
    if isinstance(l, t.nn.Embedding):
        l.weight.data = l.weight.data.float()

class Conv1D(nn.Module):
    def __init__(self, n_in, n_out, zero_out=False, init_scale=1.0):
        super(Conv1D, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if zero_out:
            w = t.zeros(n_in, n_out)
        else:
            w = t.empty(n_in, n_out)
            nn.init.normal_(w, std=0.02 * init_scale)
        b = t.zeros(n_out)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        size_out = (*x.size()[:-1], self.n_out)
        x = t.addmm(self.b.type_as(x), x.view(-1, x.size(-1)), self.w.type_as(x)) # If x if float then float else half
        x = x.view(*size_out)
        return x

# For large contexts, mask's can take up memory, so you can make a single saved mask for all layers
class Mask(nn.Module):
    def __init__(self, n_ctx):
        super().__init__()
        self.register_buffer('b', t.tril(t.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def forward(self, w):
        w = w * self.b + -1e9 * (1 - self.b)  # For fp16 do w = w.float().masked_fill(self.b, float('-inf')
        return w

def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    #assert logits.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))  # Safety check
    assert (top_k == 0) or (top_p == 0.0)
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < t.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = t.sort(logits, descending=True, dim=-1)
        cumulative_probs = t.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        #indices_to_remove = sorted_indices[sorted_indices_to_remove]
        indices_to_remove = t.zeros_like(logits, dtype=t.uint8).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
