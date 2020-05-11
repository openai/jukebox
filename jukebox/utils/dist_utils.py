import os
from time import sleep
import torch
import jukebox.utils.dist_adapter as dist

def print_once(msg):
    if (not dist.is_available()) or dist.get_rank()==0:
        print(msg)

def print_all(msg):
    if (not dist.is_available()):
        print(msg)
    elif dist.get_rank()%8==0:
        print(f'{dist.get_rank()//8}: {msg}')

def allgather(x):
    xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x)
    xs = torch.cat(xs, dim=0)
    return xs

def allreduce(x, op=dist.ReduceOp.SUM):
    x = torch.tensor(x).float().cuda()
    dist.all_reduce(x, op=op)
    return x.item()

def allgather_lists(xs):
    bs = len(xs)
    total_bs = dist.get_world_size()*len(xs)
    lengths = torch.tensor([len(x) for x in xs], dtype=t.long, device='cuda')
    lengths = allgather(lengths)
    assert lengths.shape == (total_bs,)
    max_length = torch.max(lengths).item()

    xs = torch.tensor([[*x, *[0]*(max_length - len(x))] for x in xs], device='cuda')
    assert xs.shape == (bs, max_length), f'Expected {(bs, max_length)}, got {xs.shape}'
    xs = allgather(xs)
    assert xs.shape == (total_bs,max_length), f'Expected {(total_bs, max_length)}, got {xs.shape}'

    return [xs[i][:lengths[i]].cpu().numpy().tolist() for i in range(total_bs)]

def setup_dist_from_mpi(
    master_addr="127.0.0.1", backend="nccl", port=29500, n_attempts=5, verbose=False
):
    if dist.is_available():
        return _setup_dist_from_mpi(master_addr, backend, port, n_attempts, verbose)
    else:
        use_cuda = torch.cuda.is_available()
        print(f'Using cuda {use_cuda}')

        mpi_rank = 0
        local_rank = 0

        device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
        torch.cuda.set_device(local_rank)

        return mpi_rank, local_rank, device

def _setup_dist_from_mpi(master_addr, backend, port, n_attempts, verbose):
    from mpi4py import MPI  # This must be imported in order to get e   rrors from all ranks to show up

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()


    os.environ["RANK"] = str(mpi_rank)
    os.environ["WORLD_SIZE"] = str(mpi_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "2"
    os.environ["NCCL_SOCKET_NTHREADS"] = "8"

    # Pin this rank to a specific GPU on the node
    local_rank = mpi_rank % 8
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if verbose:
        print(f"Connecting to master_addr: {master_addr}")

    # There is a race condition when initializing NCCL with a large number of ranks (e.g 500 ranks)
    # We guard against the failure and then retry
    for attempt_idx in range(n_attempts):
        try:
            dist.init_process_group(backend=backend, init_method=f"env://")
            assert dist.get_rank() == mpi_rank

            use_cuda = torch.cuda.is_available()
            print(f'Using cuda {use_cuda}')
            local_rank = mpi_rank % 8
            device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
            torch.cuda.set_device(local_rank)

            return mpi_rank, local_rank, device
        except RuntimeError as e:
            print(f"Caught error during NCCL init (attempt {attempt_idx} of {n_attempts}): {e}")
            sleep(1 + (0.01 * mpi_rank))  # Sleep to avoid thundering herd
            pass

    raise RuntimeError("Failed to initialize NCCL")
