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
"""A binary to train pruned CIFAR-10 using a single GPU.

Accuracy:
vgg_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by vgg_eval.py when target sparsity in
vgg_pruning_spec.pbtxt is set to zero

Results:
Sparsity | Accuracy after 150K steps
-------- | -------------------------
0%       | 86%
50%      | 86%
75%      | TODO(suyoggupta)
90%      | TODO(suyoggupta)
95%      | 77%

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import sys
import time
import numpy as np


import tensorflow as tf

import  vgg_pruning as vgg
import pruning
from datasets import imagenet
slim = tf.contrib.slim

FLAGS = None


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    images, labels = vgg.distorted_inputs()
    # dataset = imagenet.get_split('train', './tmp/vgg_data')
    #
    # # Creates a TF-Slim DataProvider which reads the dataset in the background
    # # during both training and testing.
    # provider = slim.dataset_data_provider.DatasetDataProvider(dataset,num_readers=10,shuffle=True)
    # image, label = provider.get(['image', 'label'])
    # # batch up some training data
    # images, labels = tf.train.batch([image, label],
    #                                   batch_size=LeNet.BATCH_SIZE)
    print (images.shape)


    images = tf.cast(images, tf.float32)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = vgg.inference(images)

    print ("logits shape:", logits.shape)
    # Calculate loss.
    print ("label shape", labels.shape)
    # Calculate loss.
    loss = vgg.loss(logits, labels)


    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = vgg.train(loss, global_step)

    # Parse pruning hyperparameters
    pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)

    # Create a pruning object using the pruning hyperparameters
    pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)

    # Use the pruning_obj to add ops to the training graph to update the masks
    # The conditional_mask_update_op will update the masks only when the
    # training step is in [begin_pruning_step, end_pruning_step] specified in
    # the pruning spec proto
    mask_update_op = pruning_obj.conditional_mask_update_op()

    # Use the pruning_obj to add summaries to the graph to track the sparsity
    # of each of the layers
    pruning_obj.add_pruning_summaries()


    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = 128
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.datetime.now(), self._step, loss_value,
                              examples_per_sec, sec_per_batch))

    list_var_names = [  'vgg_16/conv1/conv1_1/biases',
                    	'vgg_16/conv1/conv1_1/weights',
                    	'vgg_16/conv1/conv1_2/biases',
                    	'vgg_16/conv1/conv1_2/weights',
                    	'vgg_16/conv2/conv2_1/biases',
                    	'vgg_16/conv2/conv2_1/weights',
                    	'vgg_16/conv2/conv2_2/biases',
                    	'vgg_16/conv2/conv2_2/weights',
                    	'vgg_16/conv3/conv3_1/biases',
                    	'vgg_16/conv3/conv3_1/weights',
                    	'vgg_16/conv3/conv3_2/biases',
                    	'vgg_16/conv3/conv3_2/weights',
                    	'vgg_16/conv3/conv3_3/biases',
                    	'vgg_16/conv3/conv3_3/weights',
                    	'vgg_16/conv4/conv4_1/biases',
                    	'vgg_16/conv4/conv4_1/weights',
                    	'vgg_16/conv4/conv4_2/biases',
                    	'vgg_16/conv4/conv4_2/weights',
                    	'vgg_16/conv4/conv4_3/biases',
                    	'vgg_16/conv4/conv4_3/weights',
                    	'vgg_16/conv5/conv5_1/biases',
                    	'vgg_16/conv5/conv5_1/weights',
                    	'vgg_16/conv5/conv5_2/biases',
                    	'vgg_16/conv5/conv5_2/weights',
                    	'vgg_16/conv5/conv5_3/biases',
                    	'vgg_16/conv5/conv5_3/weights',
                    	'vgg_16/fc6/biases',
                    	'vgg_16/fc6/weights',
                    	'vgg_16/fc7/biases',
                    	'vgg_16/fc7/weights',
                    	'vgg_16/fc8/biases',
                    	'vgg_16/fc8/weights']

    var_list_to_restore = []
    for name in list_var_names:
        var_list_to_restore = var_list_to_restore + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)

    saver = tf.train.Saver(var_list_to_restore)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      # saver.restore(mon_sess,"vgg_16.ckpt")
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
        # Update the masks
        mon_sess.run(mask_update_op)


def main(argv=None):  # pylint: disable=unused-argument
  vgg.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./tmp/vgg_train',
      help='Directory where to write event logs and checkpoint.')
  parser.add_argument(
      '--pruning_hparams',
      type=str,
      default='',
      help="""Comma separated list of pruning-related hyperparameters""")
  parser.add_argument(
      '--max_steps',
      type=int,
      default=1000000,
      help='Number of batches to run.')
  parser.add_argument(
      '--log_device_placement',
      type=bool,
      default=False,
      help='Whether to log device placement.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
