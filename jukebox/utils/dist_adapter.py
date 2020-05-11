import torch.distributed as dist
from enum import Enum

class ReduceOp(Enum):
    SUM = 0,
    PRODUCT = 1,
    MIN = 2,
    MAX = 3

    def ToDistOp(self):
        return {
            self.SUM: dist.ReduceOp.SUM,
            self.PRODUCT: dist.ReduceOp.PRODUCT,
            self.MIN: dist.ReduceOp.MIN,
            self.MAX: dist.ReduceOp.MAX
        }[self]

def is_available():
    return dist.is_available()

def get_rank():
    if is_available():
        return _get_rank()
    else:
        return 0

def get_world_size():
    if is_available():
        return _get_world_size()
    else:
        return 1

def barrier():
    if is_available():
        return _barrier()
    #else: do nothing

def all_gather(tensor_list, tensor):
    if is_available():
        return _all_gather(tensor_list, tensor)
    else:
        tensor_list[0] = tensor

def all_reduce(tensor, op=ReduceOp.SUM):
    if is_available():
        return _all_reduce(tensor, op)
    #else: do nothing

def reduce(tensor, dst, op=ReduceOp.SUM):
    if is_available():
        return _reduce(tensor, dst, op)
    #else: do nothing

def broadcast(tensor, src):
    if is_available():
        return _broadcast(tensor, src)
    #else: do nothing

def init_process_group(backend, init_method):
    if is_available():
        return _init_process_group(backend, init_method)
    #else: do nothing

def _get_rank():
    return dist.get_rank()

def _barrier():
    return dist.barrier()

def _get_world_size():
    return dist.get_world_size()

def _all_gather(tensor_list, tensor):
    return dist.all_gather(tensor_list, tensor)

def _all_reduce(tensor, op):
    return dist.all_reduce(tensor, op.ToDistOp())

def _reduce(tensor, dst, op):
    return dist.reduce(tensor, dst, op.ToDistOp())

def _broadcast(tensor, src):
    return dist.broadcast(tensor, src)

def _init_process_group(backend, init_method):
    return dist.init_process_group(backend, init_method)