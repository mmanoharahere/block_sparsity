# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Builds the CIFAR-10 network with additional variables to support pruning.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import vgg_input
import pruning

slim = tf.contrib.slim

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = vgg_input.IMAGE_SIZE
NUM_CLASSES = vgg_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = vgg_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN  # pylint: disable=line-too-long
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = vgg_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE = 32
DATA_DIR = './tmp/vgg_data'

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 2#350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.94  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(
                             stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not DATA_DIR:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
  images, labels = vgg_input.distorted_inputs(
      data_dir=data_dir, batch_size=BATCH_SIZE)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not DATA_DIR:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(DATA_DIR, 'cifar-10-batches-bin')
  images, labels = vgg_input.inputs(
      eval_data=eval_data, data_dir=data_dir, batch_size=BATCH_SIZE)
  return images, labels


def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # While instantiating conv and local layers, we add mask and threshold
  # variables to the layer by calling the pruning.apply_mask() function.
  # Note that the masks are applied only to the weight tensors
  # conv1
  with tf.variable_scope('vgg_16') as scope:
    with tf.variable_scope('conv1') as scope:
        with tf.variable_scope('conv1_1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                images, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv1_1)

        with tf.variable_scope('conv1_2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv1_1, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv1_2)

        # pool1
        pool1 = tf.nn.max_pool(
          conv1_2,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
          name='pool1')

    with tf.variable_scope('conv2') as scope:
        with tf.variable_scope('conv2_1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool1, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv2_1)

        with tf.variable_scope('conv2_2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv2_1, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv2_2)

        # pool2
        pool2 = tf.nn.max_pool(
          conv2_2,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
          name='pool2')

    with tf.variable_scope('conv3') as scope:
        with tf.variable_scope('conv3_1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool2, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv3_1)

        with tf.variable_scope('conv3_2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv3_1, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv3_2)

        with tf.variable_scope('conv3_3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv3_2, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv3_3)

        # pool3
        pool3 = tf.nn.max_pool(
          conv3_3,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
          name='pool3')

    with tf.variable_scope('conv4') as scope:
        with tf.variable_scope('conv4_1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool3, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv4_1)

        with tf.variable_scope('conv4_2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv4_1, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv4_2)

        with tf.variable_scope('conv4_3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv4_2, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv4_3)

        # pool4
        pool4 = tf.nn.max_pool(
          conv4_3,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
          name='pool3')

    with tf.variable_scope('conv5') as scope:
        with tf.variable_scope('conv5_1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool4, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv5_1)

        with tf.variable_scope('conv5_2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv5_1, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv5_2)

        with tf.variable_scope('conv5_3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv5_2, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.tanh(pre_activation, name=scope.name)
            _activation_summary(conv5_3)

        # pool5
        pool4 = tf.nn.max_pool(
          conv5_3,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
          name='pool3')


  # fc6
    with tf.variable_scope('fc6') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool4, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', shape=[7, 7, 512, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
        weights2 = tf.reshape(weights, [dim, 4096]) 
        fc6 = tf.nn.tanh(
            tf.matmul(reshape, pruning.apply_mask(weights2, scope)) + biases,
            name=scope.name)
        _activation_summary(fc6)

    fc6 = slim.dropout(fc6, 0.5, is_training=True)

  # fc7
    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[1, 1, 4096, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
        weights2 = tf.reshape(weights, [4096, 4096])
        fc7 = tf.nn.tanh(
            tf.matmul(fc6, pruning.apply_mask(weights2, scope)) + biases,
            name=scope.name)
        _activation_summary(fc7)

    fc7 = slim.dropout(fc7, 0.5, is_training=True)

  #fc8
  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
    with tf.variable_scope('fc8') as scope:
        weights = _variable_with_weight_decay(
            'weights', [1, 1, 4096, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        weights2 = tf.reshape(weights, [4096, NUM_CLASSES])
        softmax_linear = tf.add(
            tf.matmul(fc7, pruning.apply_mask(weights2, scope)),
            biases,
            name=scope.name)
        _activation_summary(softmax_linear)



  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9,name='None')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  # for l in losses + [total_loss]:
  #   # Name each loss as '(raw)' and name the moving average version of the loss
  #   # as the original loss name.
  #   tf.summary.scalar(l.op.name + ' (raw)', l)
  #   tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(
      INITIAL_LEARNING_RATE,
      global_step,
      decay_steps,
      LEARNING_RATE_DECAY_FACTOR,
      staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    # opt = tf.train.AdamOptimizer(
    #     lr,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=1.0)
    #opt=tf.train.FtrlOptimizer(
    #    lr,
    #    learning_rate_power=-0.5,
    #    initial_accumulator_value=0.1,
    #    l1_regularization_strength=0.01,
    #    l2_regularization_strength=0.01)
    # opt=optimizer = tf.train.MomentumOptimizer(
    #     lr,
    #     momentum=0.9,
    #     name='Momentum')
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                        global_step,name='GradientDescent')
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = DATA_DIR
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(dest_directory)
