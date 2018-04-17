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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import sys
import time
import numpy as np
import os
import gzip
from six.moves import urllib

import tensorflow as tf

import  AlexNet_pruning as AlexNet
import pruning
from datasets import mnist
slim = tf.contrib.slim

FLAGS = None





def train():
  """Train AlexNet for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for AlexNet.
    # images, labels = AlexNet.distorted_inputs()
    # Selects the 'validation' dataset.
    dataset = mnist.get_split('train', './tmp/AlexNet_data')

    # Creates a TF-Slim DataProvider which reads the dataset in the background
    # during both training and testing.
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,num_readers=10,shuffle=True)
    image, label = provider.get(['image', 'label'])
    # batch up some training data
    images, labels = tf.train.batch([image, label],
                                      batch_size=AlexNet.BATCH_SIZE)
    print (images.shape)


    # Build a Graph that computes the logits predictions from the
    # inference model.
    images = tf.cast(images, tf.float32)
    # images=tf.transpose(images,[1,2,3,0])#tf.reshape(images,[28,28,1,64])
    print (images.shape)
    # labels=tf.reshape(labels,[128,])
    # print (images.shape)
    logits = AlexNet.inference(images)
    print ("logits shape:", logits.shape)
    # Calculate loss.
    print ("label shape", labels.shape)
    loss = AlexNet.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = AlexNet.train(loss, global_step)

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

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
        # Update the masks
        mon_sess.run(mask_update_op)


def main(argv=None):  # pylint: disable=unused-argument
  # AlexNet.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./tmp/AlexNet_train',
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
