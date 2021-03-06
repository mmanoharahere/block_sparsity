# Model pruning: Training tensorflow models to have masked connections

This document describes the API that facilitates magnitude-based pruning of
neural network's weight tensors. The API helps inject necessary tensorflow op
into the training graph so the model can be pruned while it is being trained.

### Run with virtualenv
```bash
# Install
sudo apt-get install python python-virtualenv
virtualenv --system-site-packages ./
source ./bin/activate
easy_install -U pip
pip install --upgrade tensorflow
python -c 'import tensorflow as tf; print(tf.__version__)

# Activate
source ./bin/activate
```

### Model creation

The first step involves adding mask and threshold variables to the layers that
need to undergo pruning. The variable mask is the same shape as the layer's
weight tensor and determines which of the weights participate in the forward
execution of the graph. This can be achieved by wrapping the weight tensor of
the layer with the `apply_mask` function provided in
[pruning.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning/python/pruning.py).
For example:

```python
conv = tf.nn.conv2d(images, pruning.apply_mask(weights), stride, padding)
```

This creates a convolutional layer with additional variables mask and threshold
as shown below: ![Convolutional layer with mask and
threshold](https://storage.googleapis.com/download.tensorflow.org/example_images/mask.png "Convolutional layer with mask and threshold")

Alternatively, the API also provides variant of tensorflow layers with these
auxiliary variables built-in (see
[layers](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning/python/layers))
. Layers currently supported:

*   [layers.masked_conv2d](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning/python/layers/layers.py?l=83)

*   [layers.masked_fully_connected](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning/python/layers/layers.py?l=241)

*   [rnn_cells.MaskedLSTMCell](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning/python/layers/rnn_cells.py?l=154)

### Adding pruning ops to the training graph

The pruning library allows for specification of the following hyper parameters:

|Hyperparameter               | Type    | Default       | Description |
|:----------------------------|:-------:|:-------------:|:--------------|
| name | string | model_pruning | Name of the pruning specification. Used for adding summaries and ops under a common tensorflow name_scope |
| begin_pruning_step | integer | 0 | The global step at which to begin pruning |
| end_pruning_step   | integer | -1 | The global step at which to terminate pruning. Defaults to -1 implying that pruning continues till  the training stops |
| do_not_prune | list of strings | [""] | list of layers names that are not pruned |
| threshold_decay | float | 0.9 | The decay factor to use for exponential decay of the thresholds |
| pruning_frequency | integer | 10 | How often should the masks be updated? (in # of global_steps) |
| nbins | integer | 256 | Number of bins to use for histogram computation |
| block_height|integer | 1 | Number of rows in a block for block sparse matrices|
| block_width |integer | 1 | Number of cols in a block for block sparse matrices|
| block_pooling_function| string | AVG | The function to use to pool weight values in a block: average (AVG) or max (MAX)|
| initial_sparsity | float | 0.0 | Initial sparsity value |
| target_sparsity | float | 0.5 | Target sparsity value |
| sparsity_function_begin_step | integer | 0 | The global step at this which the gradual sparsity function begins to take effect |
| sparsity_function_end_step | integer | 100 | The global step used as the end point for the gradual sparsity function |
| sparsity_function_exponent | float | 3.0 | exponent = 1 is linearly varying sparsity between initial and final. exponent > 1 varies more slowly towards the end than the beginning |

The sparsity $$s_t$$ at global step $$t$$ is given by:

$$ s_{t}=s_{f}+\left(s_{i}-s_{f}\right)\left(1-\frac{t-t_{0}}{n\Delta t}\right)^{3} $$

The interval between sparsity_function_begin_step and sparsity_function_end_step
is divided into $$n$$ intervals of size equal to the pruning_frequency ($$\Delta
t$$). $$s_f$$ is the target_sparsity, $$s_i$$ is the initial_sparsity, $$t_0$$
is the sparsity_function_begin_step. In this equation, the
sparsity_function_exponent is set to 3.
### Adding pruning ops to the training graph

The final step involves adding ops to the training graph that monitors the
distribution of the layer's weight magnitudes and determines the layer threshold
such masking all the weights below this threshold achieves the sparsity level
desired for the current training step. This can be achieved as follows:

```python
tf.app.flags.DEFINE_string(
    'pruning_hparams', '',
    """Comma separated list of pruning-related hyperparameters""")

with tf.graph.as_default():

  # Create global step variable
  global_step = tf.train.get_global_step()

  # Parse pruning hyperparameters
  pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)

  # Create a pruning object using the pruning specification
  p = pruning.Pruning(pruning_hparams, global_step=global_step)

  # Add conditional mask update op. Executing this op will update all
  # the masks in the graph if the current global step is in the range
  # [begin_pruning_step, end_pruning_step] as specified by the pruning spec
  mask_update_op = p.conditional_mask_update_op()

  # Add summaries to keep track of the sparsity in different layers during training
  p.add_pruning_summaries()

  with tf.train.MonitoredTrainingSession(...) as mon_sess:
    # Run the usual training op in the tf session
    mon_sess.run(train_op)

    # Update the masks by running the mask_update_op
    mon_sess.run(mask_update_op)

```

## Example: Pruning and training deep CNNs on the cifar10 dataset

Please see https://www.tensorflow.org/tutorials/deep_cnn for details on neural
network architecture, setting up inputs etc. The additional changes needed to
incorporate pruning are captured in the following:

*   [cifar10_pruning.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning/examples/cifar10/cifar10_pruning.py)
    creates a deep CNN with the same architecture, but adds mask and threshold
    variables for each of the weight tensors in the convolutional and
    locally-connected layers.

*   [cifar10_train.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning/examples/cifar10/cifar10_train.py)
    add pruning ops to the training graph as described above.

To train the pruned version of cifar10:

```bash
$ examples_dir=contrib/model_pruning/examples
$ bazel build -c opt $examples_dir/cifar10:cifar10_{train,eval}
$ bazel-bin/$examples_dir/cifar10/cifar10_train --pruning_hparams=name=cifar10_pruning,begin_pruning_step=10000,end_pruning_step=100000,target_sparsity=0.9,sparsity_function_begin_step=10000,sparsity_function_end_step=100000
```

Example of training without bazel:

```shell
$ python cifar10_train.py --train_dir=tmp/cifar10_train/test/ --max_steps=50000 --pruning_hparams=name=cifar10_pruning,begin_pruning_step=1,end_pruning_step=10000,target_sparsity=0.9,sparsity_function_begin_step=1,sparsity_function_end_step=10000,block_height=8,block_width=32
```

Eval:

```shell
$ bazel-bin/$examples_dir/cifar10/cifar10_eval --run_once
```
Example of evaluation without bazel
```shell
$ python vgg_eval.py --eval_dir ./vgg16_train_17apr_8x8/ --eval_data /data/ramyadML/TF-slim-data/imageNet/processed/ --checkpoint_dir ./vgg16_train_17apr_8x8/

$ python cifar10_eval.py --eval_dir=tmp/cifar10_eval/ --eval_data=tmp/cifar10_data/ --checkpoint_dir=tmp/cifar10_train/cifar10_train_17apr_8x8/model.ckpt-98462 --run_once=RUN_ONCE

$ python LeNet_eval.py --eval_dir=tmp/LeNet_eval/ --eval_data=tmp/LeNet_data/ --checkpoint_dir=tmp/LeNet_train/model.ckpt-3308 --run_once=RUN_ONCE

```


### Block Sparsity

For some hardware architectures, it may be beneficial to induce spatially correlated sparsity. To train models in which the weight tensors have block sparse structure, set *block_height* and *block_width* hyperparameters to the desired block configuration (2x2, 4x4, 4x1, 1x8, etc). Currently, block sparsity is only supported for weight tensors which can be squeezed to rank 2. The matrix is partitioned into non-overlapping blocks of size *[block_height, block_dim]* and the either the average or max absolute value in this block is taken as a proxy for the entire block (set by *block_pooling_function* hyperparameter).
The convolution layer tensors are always pruned used block dimensions of [1,1].

## References

Michael Zhu and Suyog Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression”, *2017 NIPS Workshop on Machine Learning of Phones and other Consumer Devices* (https://arxiv.org/pdf/1710.01878.pdf)
