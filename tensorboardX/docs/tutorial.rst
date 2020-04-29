Tutorials
*********

What is tensorboard X?
----------------------

At first, the package was named tensorboard, and soon there are issues about
name confliction. The first alternative name came to my mind is
tensorboard-pytorch, but in order to make it more general, I chose tensorboardX
which stands for tensorboard for X.

Google's tensorflow's tensorboard is a web server to serve visualizations of the
training progress of a neural network, it visualizes scalar values, images,
text, etc.; these information are saved as events in tensorflow. It's a pity
that other deep learning frameworks lack of such tool, so there are already
packages letting users to log the events without tensorflow; however they only
provides basic functionalities. The purpose of this package is to let
researchers use a simple interface to log events within PyTorch (and then show
visualization in tensorboard). This package currently supports logging scalar,
image, audio, histogram, text, embedding, and the route of back-propagation. The
following manual is tested on Ubuntu and Mac, and the environment are anaconda's
python2 and python3.


Create a summary writer
-----------------------
Before logging anything, we need to create a writer instance. This can be done with:

.. code-block:: python

    from tensorboardX import SummaryWriter
    #SummaryWriter encapsulates everything
    writer = SummaryWriter('runs/exp-1')
    #creates writer object. The log will be saved in 'runs/exp-1'
    writer2 = SummaryWriter()
    #creates writer2 object with auto generated file name, the dir will be something like 'runs/Aug20-17-20-33'
    writer3 = SummaryWriter(comment='3x learning rate')
    #creates writer3 object with auto generated file name, the comment will be appended to the filename. The dir will be something like 'runs/Aug20-17-20-33-3xlearning rate'

Each subfolder will be treated as different experiments in tensorboard. Each
time you re-run the experiment with different settings, you should change the
name of the sub folder such as ``runs/exp2``, ``runs/myexp`` so that you can
easily compare different experiment settings. Type ``tensorboard runs`` to compare
different runs in tensorboard.


General api format
------------------
.. code-block:: python

    add_something(tag name, object, iteration number)


Add scalar
-----------
Scalar value is the most simple data type to deal with. Mostly we save the loss
value of each training step, or the accuracy after each epoch. Sometimes I save
the corresponding learning rate as well. It's cheap to save scalar value. Just
log anything you think is important. To log a scalar value, use
``writer.add_scalar('myscalar', value, iteration)``. Note that the program complains
if you feed a PyTorch tensor. Remember to extract the scalar value by
``x.item()`` if ``x`` is a torch scalar tensor.


Add image
---------
An image is represented as 3-dimensional tensor. The simplest case is save one
image at a time. In this case, the image should be passed as a 3-dimension
tensor of size ``[3, H, W]``. The three dimensions correspond to R, G, B channel of
an image. After your image is computed, use ``writer.add_image('imresult', x,
iteration)`` to save the image. If you have a batch of images to show, use
``torchvision``'s ``make_grid`` function to prepare the image array and send the result
to ``add_image(...)`` (``make_grid`` takes a 4D tensor and returns tiled images in 3D tensor).

.. Note::
	Remember to normalize your image.


Add histogram
-------------
Saving histograms is expensive. Both in computation time and storage. If training
slows down after using this package, check this first. To save a histogram,
convert the array into numpy array and save with ``writer.add_histogram('hist',
array, iteration)``.


Add figure
----------
You can save a matplotlib figure to tensorboard with the add_figure function. ``figure`` input should be ``matplotlib.pyplot.figure`` or a list of ``matplotlib.pyplot.figure``.
Check `<https://tensorboardx.readthedocs.io/en/latest/tensorboard.html#tensorboardX.SummaryWriter.add_figure>`_ for the detailed usage.

Add graph
---------
To visualize a model, you need a model ``m`` and the input ``t``. ``t`` can be a tensor or a list of tensors
depending on your model. If error happens, make sure that ``m(t)`` runs without problem first. See
`The graph demo <https://github.com/lanpa/tensorboardX/blob/master/examples/demo_graph.py>`_ for
complete example.


Add audio
---------
To log a single channel audio, use ``add_audio(tag, audio, iteration, sample_rate)``, where ``audio`` is an one dimensional array, and each element in the array represents the consecutive amplitude samples.
For a 2 seconds audio with ``sample_rate`` 44100 Hz, the input ``x`` should have 88200 elements.
Each element should lie in [−1, 1].

Add embedding
-------------
Embeddings, high dimensional data, can be visualized and converted
into human perceptible 3D data by tensorboard, which provides PCA and
t-sne to project the data into low dimensional space. What you need to do is
provide a bunch of points and tensorboard will do the rest for you. The bunch of
points is passed as a tensor of size ``n x d``, where ``n`` is the number of points and
``d`` is the feature dimension. The feature representation can either be raw data
(*e.g.* the MNIST image) or a representation learned by your network (extracted
feature). This determines how the points distributes. To make the visualization
more informative, you can pass optional metadata or ``label_imgs`` for each data
points. In this way you can see that neighboring point have similar label and
distant points have very different label (semantically or visually). Here the
metadata is a list of labels, and the length of the list should equal to ``n``, the
number of the points. The ``label_imgs`` is a 4D tensor of size ``NCHW``. ``N`` should equal
to ``n`` as well. See
`The embedding demo <https://github.com/lanpa/tensorboardX/blob/master/examples/demo_embedding.py>`_ for
complete example.


Useful commands
---------------
Install
=======

Simply type ``pip install tensorboardX`` in a unix shell to install this package.
To use the newest version, you might need to build from source or ``pip install
tensorboardX —-no-cache-dir`` .  To run tensorboard web server, you need
to install it using ``pip install tensorboard``.
After that, type ``tensorboard --logdir=<your_log_dir>`` to start the server, where
``your_log_dir`` is the parameter of the object constructor. I think this command is
tedious, so I add a line alias ``tb='tensorboard --logdir '`` in ``~/.bashrc``. In
this way, the above command is simplified as ``tb <your_log_dir>``. Use your favorite
browser to load the tensorboard page, the address will be shown in the terminal
after starting the server.


Misc
----
Performance issue
=================
Logging is cheap, but display is expensive.
For my experience, if there are 3 or more experiments to show at a time and each
experiment have, say, 50k points, tensorboard might need a lot of time to
present the data.


Grouping plots
==============
Usually, there are many numbers to log in one experiment. For example, when
training GANs you should log the loss of the generator, discriminator. If the
loss is composed of two other loss functions, say L1 and MSE, you might want to
log the value of the other two losses as well. In this case, you can write the
tags as Gen/L1, Gen/MSE, Desc/L1, Desc/MSE. In this way, tensorboard will group
the plots into two sections (Gen, Desc). You can also use the regular expression
to filter data.
