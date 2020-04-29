import torch
from torch._utils import _flatten_dense_tensors
import numpy as np

# EMA always in float, as accumulation needs lots of bits
class EMA:
    def __init__(self, params, mu=0.999):
        self.mu = mu
        self.state = [(p, self.get_model_state(p)) for p in params if p.requires_grad]

    def get_model_state(self, p):
        return p.data.float().detach().clone()

    def step(self):
        for p, state in self.state:
            state.mul_(self.mu).add_(1 - self.mu, p.data.float())

    def swap(self):
        # swap ema and model params
        for p, state in self.state:
            other_state = self.get_model_state(p)
            p.data.copy_(state.type_as(p.data))
            state.copy_(other_state)


class CPUEMA:
    def __init__(self, params, mu=0.999, freq=1):
        self.mu = mu**freq
        self.state = [(p, self.get_model_state(p)) for p in params if p.requires_grad]
        self.freq = freq
        self.steps = 0

    def get_model_state(self, p):
        with torch.no_grad():
            state = p.data.float().detach().cpu().numpy()
        return state

    def step(self):
        with torch.no_grad():
            self.steps += 1
            if self.steps % self.freq == 0:
                for i in range(len(self.state)):
                    p, state = self.state[i]
                    state = torch.from_numpy(state).cuda()
                    state.mul_(self.mu).add_(1 - self.mu, p.data.float())
                    self.state[i] = (p, state.cpu().numpy())

    def swap(self):
        with torch.no_grad():
            # swap ema and model params
            for p, state in self.state:
                other_state = self.get_model_state(p)
                p.data.copy_(torch.from_numpy(state).type_as(p.data))
                np.copyto(state, other_state)

class FusedEMA:
    def __init__(self, params, mu=0.999):
        self.mu = mu
        params = list(params)
        self.params = {}
        self.params['fp16'] = [p for p in params if p.requires_grad and p.data.dtype == torch.float16]
        self.params['fp32'] = [p for p in params if p.requires_grad and p.data.dtype != torch.float16]
        self.groups = [group for group in self.params.keys() if len(self.params[group]) > 0]
        self.state = {}
        for group in self.groups:
            self.state[group] = self.get_model_state(group)

    def get_model_state(self, group):
        params = self.params[group]
        return _flatten_dense_tensors([p.data.float() for p in params])
        # if self.fp16:
        #     return _flatten_dense_tensors([p.data.half() for p in self.param_group if p.dtype])
        # else:
        #     return _flatten_dense_tensors([p.data for p in self.param_group])

    def step(self):
        for group in self.groups:
            self.state[group].mul_(self.mu).add_(1 - self.mu, self.get_model_state(group))

    def swap(self):
        # swap ema and model params
        for group in self.groups:
            other_state = self.get_model_state(group)
            state = self.state[group]
            params = self.params[group]
            offset = 0
            for p in params:
                numel = p.data.numel()
                p.data = state.narrow(0, offset, numel).view_as(p.data).type_as(p.data)
                offset += numel

            self.state[group] = other_state


