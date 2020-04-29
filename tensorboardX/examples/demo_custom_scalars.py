from numpy.random import rand
from tensorboardX import SummaryWriter
import time


with SummaryWriter() as writer:
    for n_iter in range(100):
        writer.add_scalar('twse/0050', rand(), n_iter)
        writer.add_scalar('twse/2330', rand(), n_iter)
        t = rand()
        writer.add_scalar('dow/aaa', t, n_iter)
        writer.add_scalar('dow/bbb', t - 1, n_iter)
        writer.add_scalar('dow/ccc', t + 1, n_iter)
        writer.add_scalar('nasdaq/aaa', rand(), n_iter)
        writer.add_scalar('nasdaq/bbb', rand(), n_iter)
        writer.add_scalar('nasdaq/ccc', rand(), n_iter)

    layout = {'Taiwan': {'twse': ['Multiline', ['twse/0050', 'twse/2330']]},
              'USA': {'dow': ['Margin', ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                      'nasdaq': ['Margin', ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}
    writer.add_custom_scalars(layout)
#    writer.add_custom_scalars(layout) second call has no effect

time.sleep(1)

with SummaryWriter() as writer:
    for n_iter in range(100):
        writer.add_scalar('twse/0050', rand(), n_iter)
        writer.add_scalar('twse/2330', rand(), n_iter)

    writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])

time.sleep(1)

with SummaryWriter() as writer:
    for n_iter in range(100):
        t = rand()
        writer.add_scalar('dow/aaa', t, n_iter)
        writer.add_scalar('dow/bbb', t - 1, n_iter)
        writer.add_scalar('dow/ccc', t + 1, n_iter)

    writer.add_custom_scalars_marginchart(['dow/aaa', 'dow/bbb', 'dow/ccc'])
