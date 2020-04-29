import matplotlib.pyplot as plt
plt.switch_backend('agg')

fig = plt.figure()

c1 = plt.Circle((0.2, 0.5), 0.2, color='r')
c2 = plt.Circle((0.8, 0.5), 0.2, color='r')

ax = plt.gca()
ax.add_patch(c1)
ax.add_patch(c2)
plt.axis('scaled')


from tensorboardX import SummaryWriter
writer = SummaryWriter()
writer.add_figure('matplotlib', fig)
writer.close()
