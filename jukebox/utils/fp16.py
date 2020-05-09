# Utils for fp16 training.
import importlib
import math
import numpy as np
import torch
import jukebox.utils.dist_adapter as dist
from torch.optim import Optimizer
from torch._utils import _flatten_dense_tensors

from jukebox.utils.dist_utils import allreduce

def adam_step(p: torch.Tensor, out_p: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, grad: torch.Tensor,
              lr: float, beta1: float, beta2: float, eps: float, scale: float, step: int, eps_mode: int, bias_correction: int, weight_decay: float):
    assert bias_correction == 1
    assert eps_mode == 1

    grad = grad.float()
    grad.div_(scale)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = exp_avg_sq.sqrt().add_(eps)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

    p.add_(exp_avg/denom + weight_decay*p.float(), alpha=-step_size)

# Import fused_adam if we have apex, otherwise use regular adam
try:
    fused_adam_cuda = importlib.import_module("fused_adam_cuda")
    fused_adam_step = fused_adam_cuda.adam
    print("Using apex fused_adam_cuda")
except ModuleNotFoundError:
    fused_adam_step = adam_step

def backward(loss, params, scalar, fp16, logger):
    # Perform backward
    if not fp16:
        scale = 1.0
        loss.backward()
        gn = grad_norm(params, scale)
        return loss, scale, gn, False, False
    else:
        scale = scalar.get_scale()
        loss = (loss.float())*scale
        overflow_loss = check_overflow(loss.item())
        overflow_loss = allreduce(int(overflow_loss), op=dist.ReduceOp.MAX) > 0
        if not overflow_loss:
            loss.backward()
            gn = grad_norm(params, scale)
            overflow_grad = check_overflow(gn)
            overflow_grad = allreduce(int(overflow_grad), op=dist.ReduceOp.MAX) > 0
            scalar.update_scale(overflow_grad)
        else:
            gn = 0.0
            overflow_grad = True
        loss = (loss.detach().float()) / scale # Should delete computation graph for overflow
        if logger.rank == 0:
            if loss > 12.: print(f"\nWarning. Loss is {loss}")
            if overflow_loss: print(f"\nOverflow in forward. Loss {loss}, lgscale {np.log2(scale)}. Skipping batch completely (no backward, scale update)")
            elif overflow_grad: print(f"\nOverflow in backward. Loss {loss}, grad norm {gn}, lgscale {np.log2(scale)}, new lgscale {np.log2(scalar.get_scale())}")
        return loss, scale, gn, overflow_loss, overflow_grad

# Automatic loss scaling
class LossScalar(object):
    def __init__(self,
                 loss_scale,
                 init_scale=2. ** 16,
                 scale_factor=2. ** (1. / 1000),
                 scale_window=1):
        if loss_scale == None:
            # Use dynamic loss scaling
            self.dynamic = True
            self.loss_scale = init_scale
        else:
            self.dynamic = False
            self.loss_scale = loss_scale
        self.max_loss_scale = 2.**24
        self.scale_factor = scale_factor
        self.scale_window  = scale_window
        self.unskipped = 0
        self.overflow = False

    def get_scale(self):
        return self.loss_scale

    def update_scale(self, overflow):
        if overflow and self.dynamic:
            self.loss_scale /= 2.
            self.unskipped = 0
        else:
            self.unskipped += 1

        if self.unskipped == self.scale_window and self.dynamic:
            self.loss_scale = min(self.max_loss_scale, self.loss_scale * self.scale_factor)
            self.unskipped = 0

def check_overflow(val):
    return (val == float('inf')) or (val == -float('inf')) or (val != val)

def grad_norm(params, scale, flat=False):
    params = list(params)
    if flat:
        # Faster but more memory
        fp16_grads = [p.grad for p in params if p.grad is not None and p.data.dtype == torch.float16]
        fp16_norm = 0.0 if len(fp16_grads) == 0 else float(_flatten_dense_tensors(fp16_grads).norm(p=2, dtype=torch.float32))
        fp32_grads = [p.grad for p in params if p.grad is not None and p.data.dtype != torch.float16]
        fp32_norm = 0.0 if len(fp32_grads) == 0 else float(_flatten_dense_tensors(fp32_grads).norm(p=2))
        grad_norm = (fp16_norm**2 + fp32_norm**2)**0.5
    else:
        # Slightly slower but less memory
        grad_norm = 0.0
        for p in params:
            if p.grad is not None:
                grad_norm += p.grad.norm(p=2, dtype=torch.float32)**2
        grad_norm = float(grad_norm**0.5)
    return grad_norm / scale

def clipped_grad_scale(grad_norm, max_grad_norm, scale):
    clip = grad_norm / max_grad_norm
    if clip > 1:
        scale = clip * scale
    return scale

class FP16FusedAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
        amsgrad=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        defaults = dict(
            lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super(FP16FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.FLOAT16_MAX = 65504.0
        self.init_state()

    def init_state(self):
        for group in self.param_groups:
            for p in group["params"]:
                assert p.requires_grad == True
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if p.data.dtype == torch.float16:
                        state["scale_exp_avg"] = 1.0
                        state["scale_exp_avg_sq"] = 1.0

    def step(self, closure=None, scale=1.0):
        """Performs a single optimization step. Scales gradients down by scale
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if p.data.dtype == torch.float16:
                    exp_avg, exp_avg_sq = (
                        state["exp_avg"].float() * state["scale_exp_avg"],
                        state["exp_avg_sq"].float() * state["scale_exp_avg_sq"],
                    )
                else:
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = torch.tensor([], dtype=torch.float)
                fused_adam_step(
                    p.data,
                    out_p,
                    exp_avg,
                    exp_avg_sq,
                    grad,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    scale,
                    state["step"],
                    self.eps_mode,
                    bias_correction,
                    group["weight_decay"],
                )

                if p.data.dtype == torch.float16:
                    state["scale_exp_avg"] = (
                        1e-8 + float(torch.norm(exp_avg, float("inf"))) / self.FLOAT16_MAX
                    )
                    state["scale_exp_avg_sq"] = (
                        1e-8 + float(torch.norm(exp_avg_sq, float("inf"))) / self.FLOAT16_MAX
                    )
                    state["exp_avg"] = (exp_avg / state["scale_exp_avg"]).half()
                    state["exp_avg_sq"] = (exp_avg_sq / state["scale_exp_avg_sq"]).half()

        return loss


class FusedAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
        amsgrad=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        defaults = dict(
            lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    def step(self, closure=None, scale=1.0):
        """Performs a single optimization step. Scales gradients down by scale
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = torch.tensor([], dtype=torch.float)
                fused_adam_step(
                    p.data,
                    out_p,
                    exp_avg,
                    exp_avg_sq,
                    grad,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    scale,
                    state["step"],
                    self.eps_mode,
                    bias_correction,
                    group["weight_decay"],
                )

        return loss

