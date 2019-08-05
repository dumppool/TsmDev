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

"""Resnet model configuration.

References:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition
  arXiv:1512.03385 (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks
  arXiv:1603.05027 (2016)

  Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy,
  Alan L. Yuille
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  arXiv:1606.00915 (2016)
"""

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import datasets
from models import model as model_lib


def bottleneck_block_v1(cnn, depth, depth_bottleneck, stride):
  """Bottleneck block with identity short-cut for ResNet v1.

  Args:
    cnn: the network to append bottleneck blocks.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
  """
  input_layer = cnn.top_layer
  in_size = cnn.top_size
  name_key = 'resnet_v1'
  name = name_key + str(cnn.counts[name_key])
  cnn.counts[name_key] += 1

  with tf.variable_scope(name):
    if depth == in_size:
      if stride == 1:
        shortcut = input_layer
      else:
        shortcut = cnn.apool(
            1, 1, stride, stride, input_layer=input_layer,
            num_channels_in=in_size)
    else:
      shortcut = cnn.conv(
          depth, 1, 1, stride, stride, activation=None,
          use_batch_norm=True, input_layer=input_layer,
          num_channels_in=in_size, bias=None)
    cnn.conv(depth_bottleneck, 1, 1, stride, stride,
             input_layer=input_layer, num_channels_in=in_size,
             use_batch_norm=True, bias=None)
    cnn.conv(depth_bottleneck, 3, 3, 1, 1, mode='SAME_RESNET',
             use_batch_norm=True, bias=None)
    res = cnn.conv(depth, 1, 1, 1, 1, activation=None,
                   use_batch_norm=True, bias=None)
    output = tf.nn.relu(shortcut + res)
    cnn.top_layer = output
    cnn.top_size = depth


def bottleneck_block(cnn, depth, depth_bottleneck, stride, pre_activation):
    bottleneck_block_v1(cnn, depth, depth_bottleneck, stride)


class ResnetModel(model_lib.Model):
  """Resnet cnn network configuration."""

  def __init__(self, model, layer_counts):
    default_batch_sizes = {
        'resnet50': 64,
        'resnet101': 32,
        'resnet152': 32,
        'resnet50_v2': 64,
        'resnet101_v2': 32,
        'resnet152_v2': 32,
    }
    batch_size = default_batch_sizes.get(model, 32)
    super(ResnetModel, self).__init__(model, 224, batch_size, 0.005,
                                      layer_counts)
    self.pre_activation = 'v2' in model

  def add_inference(self, cnn):
    if self.layer_counts is None:
      raise ValueError('Layer counts not specified for %s' % self.get_model())
    cnn.use_batch_norm = True
    cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}
    cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET', use_batch_norm=True)
    cnn.mpool(3, 3, 2, 2)
    for _ in xrange(self.layer_counts[0]):
      bottleneck_block(cnn, 256, 64, 1, self.pre_activation)
    for i in xrange(self.layer_counts[1]):
      stride = 2 if i == 0 else 1
      bottleneck_block(cnn, 512, 128, stride, self.pre_activation)
    for i in xrange(self.layer_counts[2]):
      stride = 2 if i == 0 else 1
      bottleneck_block(cnn, 1024, 256, stride, self.pre_activation)
    for i in xrange(self.layer_counts[3]):
      stride = 2 if i == 0 else 1
      bottleneck_block(cnn, 2048, 512, stride, self.pre_activation)
    if self.pre_activation:
      cnn.batch_norm()
      cnn.top_layer = tf.nn.relu(cnn.top_layer)
    cnn.spatial_mean()

  def get_learning_rate(self, global_step, batch_size):
    num_batches_per_epoch = (
        float(datasets.IMAGENET_NUM_TRAIN_IMAGES) / batch_size)
    boundaries = [int(num_batches_per_epoch * x) for x in [30, 60]]
    values = [0.1, 0.01, 0.001]
    return tf.train.piecewise_constant(global_step, boundaries, values)


def create_resnet50_model():
  return ResnetModel('resnet50', (3, 4, 6, 3))


