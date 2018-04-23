# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import numpy as np
sns.set()

np.set_printoptions(threshold=np.nan)

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = None


def print_tensors_in_checkpoint_file(file_name, tensor_name, mask, all_tensors,
                                     all_tensor_names=False):
  """Prints tensors in a checkpoint file.
  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.
  If `tensor_name` is provided, prints the content of the tensor.
  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
    all_tensor_names: Boolean indicating whether to print all tensor names.
  """
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    kernel_array =  {}#[16,[[],[],[],[]]]#np.zeros(shape=[16,])
    biases_array = {}#np.empty_like()#np.zeros(shape=[16,])
    mask_array = {}#np.empty_like()#np.zeros(shape=[16,])
    weight_loss = {}
    target_layer = [ "vgg_16/conv1/conv1_1/weights",
                     "vgg_16/conv1/conv1_2/weights",
                     "vgg_16/conv2/conv2_1/weights",
                     "vgg_16/conv2/conv2_2/weights",
                     "vgg_16/conv3/conv3_1/weights",
                     "vgg_16/conv3/conv3_2/weights",
                     "vgg_16/conv3/conv3_3/weights",
                     "vgg_16/conv4/conv4_1/weights",
                     "vgg_16/conv4/conv4_2/weights",
                     "vgg_16/conv4/conv4_3/weights",
                     "vgg_16/conv5/conv5_1/weights",
                     "vgg_16/conv5/conv5_2/weights",
                     "vgg_16/conv5/conv5_3/weights",
                     "vgg_16/fc6/weights",
                     "vgg_16/fc7/weights",
                     "vgg_16/fc8/weights",
                     "vgg_16/conv1/conv1_1/biases",
                     "vgg_16/conv1/conv1_2/biases",
                     "vgg_16/conv2/conv2_1/biases",
                     "vgg_16/conv2/conv2_2/biases",
                     "vgg_16/conv3/conv3_1/biases",
                     "vgg_16/conv3/conv3_2/biases",
                     "vgg_16/conv3/conv3_3/biases",
                     "vgg_16/conv4/conv4_1/biases",
                     "vgg_16/conv4/conv4_2/biases",
                     "vgg_16/conv4/conv4_3/biases",
                     "vgg_16/conv5/conv5_1/biases",
                     "vgg_16/conv5/conv5_2/biases",
                     "vgg_16/conv5/conv5_3/biases",
                     "vgg_16/fc6/biases",
                     "vgg_16/fc7/biases",
                     "vgg_16/fc8/biases",
                     "vgg_16/conv1/conv1_1/mask",
                     "vgg_16/conv1/conv1_2/mask",
                     "vgg_16/conv2/conv2_1/mask",
                     "vgg_16/conv2/conv2_2/mask",
                     "vgg_16/conv3/conv3_1/mask",
                     "vgg_16/conv3/conv3_2/mask",
                     "vgg_16/conv3/conv3_3/mask",
                     "vgg_16/conv4/conv4_1/mask",
                     "vgg_16/conv4/conv4_2/mask",
                     "vgg_16/conv4/conv4_3/mask",
                     "vgg_16/conv5/conv5_1/mask",
                     "vgg_16/conv5/conv5_2/mask",
                     "vgg_16/conv5/conv5_3/mask",
                     "vgg_16/fc6/mask",
                     "vgg_16/fc7/mask",
                     "vgg_16/fc8/mask",
                     "global_step",
                     "total_loss/None",
                     "Mean/None",
                     "vgg16_pruning/last_mask_update_step",
                     "vgg_16/conv1/conv1_1/weight_loss/None",
                     "vgg_16/conv1/conv1_2/weight_loss/None",
                     "vgg_16/conv2/conv2_1/weight_loss/None",
                     "vgg_16/conv2/conv2_2/weight_loss/None",
                     "vgg_16/conv3/conv3_1/weight_loss/None",
                     "vgg_16/conv3/conv3_2/weight_loss/None",
                     "vgg_16/conv3/conv3_3/weight_loss/None",
                     "vgg_16/conv4/conv4_1/weight_loss/None",
                     "vgg_16/conv4/conv4_2/weight_loss/None",
                     "vgg_16/conv4/conv4_3/weight_loss/None",
                     "vgg_16/conv5/conv5_1/weight_loss/None",
                     "vgg_16/conv5/conv5_2/weight_loss/None",
                     "vgg_16/conv5/conv5_3/weight_loss/None",
                     "vgg_16/fc6/weight_loss/None",
                     "vgg_16/fc6/weight_loss/None",
                     "vgg_16/fc8/weight_loss/None",
                     ]
    model ={
         "vgg_16/conv1/conv1_1/weights": tf.Variable(tf.truncated_normal([3,3,3,64]), name="vgg_16/conv1/conv1_1/weights"),
         "vgg_16/conv1/conv1_2/weights": tf.Variable(tf.truncated_normal([3,3,64,64]), name="vgg_16/conv1/conv1_2/weights"),
         "vgg_16/conv2/conv2_1/weights": tf.Variable(tf.truncated_normal([3,3,64,128]), name="vgg_16/conv2/conv2_1/weights"),
         "vgg_16/conv2/conv2_2/weights": tf.Variable(tf.truncated_normal([3,3,128,128]), name="vgg_16/conv2/conv2_2/weights"),
         "vgg_16/conv3/conv3_1/weights": tf.Variable(tf.truncated_normal([3,3,128,256]), name="vgg_16/conv3/conv3_1/weights"),
         "vgg_16/conv3/conv3_2/weights": tf.Variable(tf.truncated_normal([3,3,256,256]), name="vgg_16/conv3/conv3_2/weights"),
         "vgg_16/conv3/conv3_3/weights": tf.Variable(tf.truncated_normal([3,3,256,256]), name="vgg_16/conv3/conv3_3/weights"),
         "vgg_16/conv4/conv4_1/weights": tf.Variable(tf.truncated_normal([3,3,256,512]), name="vgg_16/conv4/conv4_1/weights"),
         "vgg_16/conv4/conv4_2/weights": tf.Variable(tf.truncated_normal([3,3,512,512]), name="vgg_16/conv4/conv4_2/weights"),
         "vgg_16/conv4/conv4_3/weights": tf.Variable(tf.truncated_normal([3,3,512,512]), name="vgg_16/conv4/conv4_3/weights"),
         "vgg_16/conv5/conv5_1/weights": tf.Variable(tf.truncated_normal([3,3,512,512]), name="vgg_16/conv5/conv5_1/weights"),
         "vgg_16/conv5/conv5_2/weights": tf.Variable(tf.truncated_normal([3,3,512,512]), name="vgg_16/conv5/conv5_2/weights"),
         "vgg_16/conv5/conv5_3/weights": tf.Variable(tf.truncated_normal([3,3,512,512]), name="vgg_16/conv5/conv5_3/weights"),
         "vgg_16/fc6/weights": tf.Variable(tf.truncated_normal([7,7,512, 4096]), name="vgg_16/fc6/weights"),
         "vgg_16/fc7/weights": tf.Variable(tf.truncated_normal([1, 1, 4096, 4096]), name="vgg_16/fc7/weights"),
         "vgg_16/fc8/weights": tf.Variable(tf.truncated_normal([1, 1, 4096, 1000]), name="vgg_16/fc8/weights"),
         "vgg_16/conv1/conv1_1/biases": tf.Variable(tf.truncated_normal([64]), name="vgg_16/conv1/conv1_1/biases"),
         "vgg_16/conv1/conv1_2/biases": tf.Variable(tf.truncated_normal([64]), name="vgg_16/conv1/conv1_2/biases"),
         "vgg_16/conv2/conv2_1/biases": tf.Variable(tf.truncated_normal([128]), name="vgg_16/conv2/conv2_1/biases"),
         "vgg_16/conv2/conv2_2/biases": tf.Variable(tf.truncated_normal([128]), name="vgg_16/conv2/conv2_2/biases"),
         "vgg_16/conv3/conv3_1/biases": tf.Variable(tf.truncated_normal([256]), name="vgg_16/conv3/conv3_1/biases"),
         "vgg_16/conv3/conv3_2/biases": tf.Variable(tf.truncated_normal([256]), name="vgg_16/conv3/conv3_2/biases"),
         "vgg_16/conv3/conv3_3/biases": tf.Variable(tf.truncated_normal([256]), name="vgg_16/conv3/conv3_3/biases"),
         "vgg_16/conv4/conv4_1/biases": tf.Variable(tf.truncated_normal([512]), name="vgg_16/conv4/conv4_1/biases"),
         "vgg_16/conv4/conv4_2/biases": tf.Variable(tf.truncated_normal([512]), name="vgg_16/conv4/conv4_2/biases"),
         "vgg_16/conv4/conv4_3/biases": tf.Variable(tf.truncated_normal([512]), name="vgg_16/conv4/conv4_3/biases"),
         "vgg_16/conv5/conv5_1/biases": tf.Variable(tf.truncated_normal([512]), name="vgg_16/conv5/conv5_1/biases"),
         "vgg_16/conv5/conv5_2/biases": tf.Variable(tf.truncated_normal([512]), name="vgg_16/conv5/conv5_2/biases"),
         "vgg_16/conv5/conv5_3/biases": tf.Variable(tf.truncated_normal([512]), name="vgg_16/conv5/conv5_3/biases"),
         "vgg_16/fc6/biases": tf.Variable(tf.truncated_normal([4096]), name="vgg_16/fc6/biases"),
         "vgg_16/fc7/biases": tf.Variable(tf.truncated_normal([4096]), name="vgg_16/fc7/biases"),
         "vgg_16/fc8/biases": tf.Variable(tf.truncated_normal([1000]), name="vgg_16/fc8/biases"),
         "global_step": tf.Variable([],name="global_step"),
         "total_loss/None": tf.Variable([],name="total_loss/None"),
         "Mean/None": tf.Variable([],name="Mean/None"),
         "vgg16_pruning/last_mask_update_step": tf.Variable([],name="vgg16_pruning/last_mask_update_step"),
         "vgg_16/conv1/conv1_1/weight_loss/None": tf.Variable([],name="vgg_16/conv1/conv1_1/weight_loss/None"),
         "vgg_16/conv1/conv1_2/weight_loss/None": tf.Variable([],name="vgg_16/conv1/conv1_2/weight_loss/None"),
         "vgg_16/conv2/conv2_1/weight_loss/None": tf.Variable([],name="vgg_16/conv2/conv2_1/weight_loss/None"),
         "vgg_16/conv2/conv2_2/weight_loss/None": tf.Variable([],name="vgg_16/conv2/conv2_2/weight_loss/None"),
         "vgg_16/conv3/conv3_1/weight_loss/None": tf.Variable([],name="vgg_16/conv3/conv3_1/weight_loss/None"),
         "vgg_16/conv3/conv3_2/weight_loss/None": tf.Variable([],name="vgg_16/conv3/conv3_2/weight_loss/None"),
         "vgg_16/conv3/conv3_3/weight_loss/None": tf.Variable([],name="vgg_16/conv3/conv3_3/weight_loss/None"),
         "vgg_16/conv4/conv4_1/weight_loss/None": tf.Variable([],name="vgg_16/conv4/conv4_1/weight_loss/None"),
         "vgg_16/conv4/conv4_2/weight_loss/None": tf.Variable([],name="vgg_16/conv4/conv4_2/weight_loss/None"),
         "vgg_16/conv4/conv4_3/weight_loss/None": tf.Variable([],name="vgg_16/conv4/conv4_3/weight_loss/None"),
         "vgg_16/conv5/conv5_1/weight_loss/None": tf.Variable([],name="vgg_16/conv5/conv5_1/weight_loss/None"),
         "vgg_16/conv5/conv5_2/weight_loss/None": tf.Variable([],name="vgg_16/conv5/conv5_2/weight_loss/None"),
         "vgg_16/conv5/conv5_3/weight_loss/None": tf.Variable([],name="vgg_16/conv5/conv5_3/weight_loss/None"),
         "vgg_16/fc6/weight_loss/None": tf.Variable([],name="vgg_16/fc6/weight_loss/None"),
         "vgg_16/fc7/weight_loss/None": tf.Variable([],name="vgg_16/fc7/weight_loss/None"),
         "vgg_16/fc8/weight_loss/None": tf.Variable([],name="vgg_16/fc8/weight_loss/None"),
    }
    if all_tensors or all_tensor_names:
      # var_to_shape_map = reader.get_variable_to_shape_map()
      # for key in tagersorted(var_to_shape_map):
      #     print(reader.get_tensor(key))
      # print("tensor_name: ", key)
      # if all_tensors:

      for key in target_layer:
      #     # print("tensor_name: ", key)
          if 'weights' in key:
              print("tensor_name: ", key)
              kernel_array[key] = reader.get_tensor(key)
          if 'biases' in key:
              print("tensor_name: ", key)
              biases_array[key] = reader.get_tensor(key)
          if 'mask' in key:
              print("tensor_name: ", key)
              mask_array[key] = reader.get_tensor(key)
          if 'global_step' in key:
              print("tensor_name: ", key)
              global_step=reader.get_tensor(key)
          if 'total_loss' in key:
              # print("tensor_name: ", key)
              total_loss=reader.get_tensor(key)
          if 'Mean' in key:
              print("tensor_name: ", key)
              Mean=reader.get_tensor(key)
          if 'last_mask_update_step' in key:
              print("tensor_name: ", key)
              last_mask_update_step=reader.get_tensor(key)
          if 'weight_loss' in key:
              print("tensor_name: ", key)
              weight_loss[key] = reader.get_tensor(key)

      # print (kernel_array.shape)
      # kernel_array = np.multiply(kernel_array,(mask_array))

      for key in target_layer:
          if 'weights' in key:
              if 'conv' in key:
                  a,b,c,d=key.split("/")
                  model[key]=tf.Variable(tf.constant(np.multiply(kernel_array[a+"/"+b+"/"+c+"/"+d],mask_array[a+"/"+b+"/"+c+"/mask"])),name=key)
              if 'fc' in key:
                  if 'fc6' in key:
                      a,b,c=key.split("/") 
                      model[key]=tf.Variable(tf.constant(np.multiply(kernel_array[a+"/"+b+"/"+c],mask_array[a+"/"+b+"/mask"].reshape(7,7,512,4096))),name=key)
                  else:
                      a,b,c=key.split("/")
                      model[key]=tf.Variable(tf.constant(np.multiply(kernel_array[a+"/"+b+"/"+c],mask_array[a+"/"+b+"/mask"])),name=key)
          if 'biases' in key:
              model[key]=tf.Variable(tf.constant(biases_array[key]),name=key)
          if 'mask' in key:
              model[key]=tf.Variable(tf.constant(mask_array[key]),name=key)
          if 'global_step' in key:
              model[key]=tf.Variable(tf.constant(global_step,name=key))
          if 'total_loss' in key:
              model[key]=tf.Variable(tf.constant(total_loss,name=key))
          if 'Mean' in key:
              model[key]=tf.Variable(tf.constant(Mean,name=key))
          if 'last_mask_update_step' in key:
              model[key]=tf.Variable(tf.constant(last_mask_update_step,name=key))
          if 'weight_loss' in key:
              model[key]=tf.Variable(tf.constant(weight_loss[key],name=key))


      with tf.Session() as sess:
          for var in tf.global_variables():
             if tf.is_variable_initialized(var).eval() == False:
                sess.run(tf.variables_initializer([var]))

          # Save model objects to serialized format
          final_saver = tf.train.Saver(model)
          final_saver.save(sess, "/tmp/model_ckpt_masked")



    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      kernel = reader.get_tensor(tensor_name)
      mask = reader.get_tensor(mask)
      # print (kernel.shape)
      # kernel = np.swapaxes(kernel, 0, 2)
      # print (kernel.shape)
      print ("non-zeros=",np.count_nonzero(mask))
      print ("%zeros=",1-np.count_nonzero(mask)/mask.size)
      kernel= kernel.reshape(np.prod(kernel.shape[:-1]), kernel.shape[-1])
      mask= mask.reshape(np.prod(mask.shape[:-1]), mask.shape[-1])
      print (kernel.shape)
      # df = pd.DataFrame(kernel)
      df = pd.DataFrame(np.multiply(kernel,(mask)))
      df = df.abs()
      # df = df[96:105]
      sns.heatmap(df, cmap="Blues", vmin=0.0, vmax=1.5)
      plt.show()
      print ("non-zeros=",np.count_nonzero(df))
      print ("zeros %=",1-np.count_nonzero(df)/df.size)
      # csv_file_name = tensor_name+".csv"
      # df.to_csv('vgg.csv',index=False)
      print(df)

  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in file_name for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))


def parse_numpy_printoption(kv_str):
  """Sets a single numpy printoption from a string of the form 'x=y'.
  See documentation on numpy.set_printoptions() for details about what values
  x and y can take. x can be any option listed there other than 'formatter'.
  Args:
    kv_str: A string of the form 'x=y', such as 'threshold=100000'
  Raises:
    argparse.ArgumentTypeError: If the string couldn't be used to set any
        nump printoption.
  """
  k_v_str = kv_str.split("=", 1)
  if len(k_v_str) != 2 or not k_v_str[0]:
    raise argparse.ArgumentTypeError("'%s' is not in the form k=v." % kv_str)
  k, v_str = k_v_str
  printoptions = np.get_printoptions()
  if k not in printoptions:
    raise argparse.ArgumentTypeError("'%s' is not a valid printoption." % k)
  v_type = type(printoptions[k])
  if v_type is type(None):
    raise argparse.ArgumentTypeError(
        "Setting '%s' from the command line is not supported." % k)
  try:
    v = (
        v_type(v_str)
        if v_type is not bool else flags.BooleanParser().parse(v_str))
  except ValueError as e:
    raise argparse.ArgumentTypeError(e.message)
  np.set_printoptions(**{k: v})


def main(unused_argv):
  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
          "[--tensor_name=tensor_to_print] "
          "[--mask=mask_of the layer]"
          "[--all_tensors] "
          "[--all_tensor_names] "
          "[--printoptions]")
    sys.exit(1)
  else:
    print_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.tensor_name, FLAGS.mask,
                                     FLAGS.all_tensors, FLAGS.all_tensor_names)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--file_name",
      type=str,
      default="",
      help="Checkpoint filename. "
      "Note, if using Checkpoint V2 format, file_name is the "
      "shared prefix between all files in the checkpoint.")
  parser.add_argument(
      "--tensor_name",
      type=str,
      default="",
      help="Name of the tensor to inspect")
  parser.add_argument(
      "--mask",
      type=str,
      default="",
      help="Name of the tensor to inspect")
  parser.add_argument(
      "--all_tensors",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="If True, print the names and values of all the tensors.")
  parser.add_argument(
      "--all_tensor_names",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="If True, print the names of all the tensors.")
  parser.add_argument(
      "--printoptions",
      nargs="*",
      type=parse_numpy_printoption,
      help="Argument for numpy.set_printoptions(), in the form 'k=v'.")
  FLAGS, unparsed = parser.parse_known_args()
app.run(main=main, argv=[sys.argv[0]] + unparsed)
