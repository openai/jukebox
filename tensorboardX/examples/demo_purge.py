from time import sleep
from tensorboardX import SummaryWriter

with SummaryWriter(logdir='runs/purge') as w:
    for i in range(100):
        w.add_scalar('purgetest', i, i)

sleep(1.0)

with SummaryWriter(logdir='runs/purge', purge_step=42) as w:
    # event 42~99 are removed (inclusively)
    for i in range(42, 100):
        w.add_scalar('purgetest', 42, i)
