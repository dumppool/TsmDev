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

"""Image pre-processing utilities.
"""
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.image.python.ops import distort_image_ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
#import cnn_util




class SyntheticImagePreprocessor(object):
  """Preprocessor used for images and labels."""

  def __init__(self, height, width, batch_size, num_splits,
               dtype, train, distortions, resize_method, shift_ratio,
               summary_verbosity, distort_color_in_yiq=False,
               fuse_decode_and_crop=False):
    del train, distortions, resize_method, summary_verbosity
    del distort_color_in_yiq
    del fuse_decode_and_crop
    self.batch_size = batch_size
    self.height = height
    self.width = width
    self.depth = 3
    self.dtype = dtype
    self.num_splits = num_splits
    self.shift_ratio = shift_ratio

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=-1):
    """Get synthetic image batches."""
    del subset, use_datasets, cache_data, shift_ratio
    input_shape = [self.batch_size, self.height, self.width, self.depth]
    images = tf.truncated_normal(
        input_shape,
        dtype=self.dtype,
        stddev=1e-1,
        name='synthetic_images')
    labels = tf.random_uniform(
        [self.batch_size],
        minval=0,
        maxval=dataset.num_classes - 1,
        dtype=tf.int32,
        name='synthetic_labels')
    # Note: This results in a H2D copy, but no computation
    # Note: This avoids recomputation of the random values, but still
    #         results in a H2D copy.
    images = tf.contrib.framework.local_variable(images, name='images')
    labels = tf.contrib.framework.local_variable(labels, name='labels')
    if self.num_splits == 1:
      images_splits = [images]
      labels_splits = [labels]
    else:
      images_splits = tf.split(images, self.num_splits, 0)
      labels_splits = tf.split(labels, self.num_splits, 0)
    return images_splits, labels_splits

