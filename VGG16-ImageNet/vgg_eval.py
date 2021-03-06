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
"""Evaluation for CIFAR-10.

Accuracy:
vgg_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by vgg_eval.py.

Speed:
On a single Tesla K40, vgg_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import math
import sys
import time

import numpy as np
import tensorflow as tf

import vgg_pruning as vgg
from datasets import imagenet
slim = tf.contrib.slim
from preprocessing import preprocessing_factory

FLAGS = None


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
    saver.restore(sess, FLAGS.checkpoint_dir)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/vgg_train/model.ckpt-0,
      # extract global_step from it.
    global_step = FLAGS.checkpoint_dir.split('/')[-1].split('-')[-1]
    # else:
    #   print('No checkpoint file found')
    #   return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / 128))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * 128
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    # eval_data = FLAGS.eval_data == 'test'
    # images, labels = vgg.inputs(eval_data=eval_data)
    dataset = imagenet.get_split('validation', '/data/ramyadML/TF-slim-data/imageNet/processed')

    # Creates a TF-Slim DataProvider which reads the dataset in the background
    # during both training and testing.
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=4,
                                                              common_queue_capacity=20*32,
                                                              common_queue_min=10*32,
                                                              shuffle=True)


    preprocessing_name = 'vgg_16'
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                            preprocessing_name,
                            is_training=True)

    [image, label] = provider.get(['image', 'label'])
    image = image_preprocessing_fn(image, 224, 224)
    label -= 1

    # batch up some training data
    images, labels = tf.train.batch([image, label],
                                    batch_size=32,
                                    num_threads=4,
                                    capacity=5*32)

    print (images.shape)


    images = tf.cast(images, tf.float32)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = vgg.inference(images)

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)
    top_k_op = slim.metrics.streaming_accuracy(predictions, labels)

    # Calculate predictions.
    # top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # # Restore the moving average version of the learned variables for eval.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     vgg.MOVING_AVERAGE_DECAY)
    # variables_to_restore = variable_averages.variables_to_restore()
    # saver = tf.train.Saver(variables_to_restore)

    # Save
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
                    	'vgg_16/fc8/weights',
                        'vgg_16/conv1/conv1_1/mask',
                        'vgg_16/conv1/conv1_2/mask',
                        'vgg_16/conv2/conv2_1/mask',
                        'vgg_16/conv2/conv2_2/mask',
                        'vgg_16/conv3/conv3_1/mask',
                        'vgg_16/conv3/conv3_2/mask',
                        'vgg_16/conv3/conv3_3/mask',
                        'vgg_16/conv4/conv4_1/mask',
                        'vgg_16/conv4/conv4_2/mask',
                        'vgg_16/conv4/conv4_3/mask',
                        'vgg_16/conv5/conv5_1/mask',
                        'vgg_16/conv5/conv5_2/mask',
                        'vgg_16/conv5/conv5_3/mask',
                        'vgg_16/fc6/mask',
                        'vgg_16/fc7/mask',
                        'vgg_16/fc8/mask']

    var_list_to_restore = []

    for name in list_var_names:
        var_list_to_restore = var_list_to_restore + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
    saver = tf.train.Saver(var_list_to_restore)


    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  # vgg.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--eval_dir',
      type=str,
      default='./tmp/vgg_eval',
      help='Directory where to write event logs.')
  parser.add_argument(
      '--eval_data',
      type=str,
      default='test',
      help="""Either 'test' or 'train_eval'.""")
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./tmp/vgg_train',
      help="""Directory where to read model checkpoints.""")
  parser.add_argument(
      '--eval_interval_secs',
      type=int,
      default=60 * 5,
      help='How often to run the eval.')
  parser.add_argument(
      '--num_examples',
      type=int,
      default=10000,
      help='Number of examples to run.')
  parser.add_argument(
      '--run_once',
      type=bool,
      default=False,
      help='Whether to run eval only once.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
