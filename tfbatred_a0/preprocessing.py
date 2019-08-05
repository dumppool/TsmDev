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


class Cifar10ImagePreprocessor(object):
  """Preprocessor for Cifar10 input images."""

  def __init__(self,
               height,
               width,
               batch_size,
               num_splits,
               dtype,
               train,
               distortions,
               resize_method,
               shift_ratio,
               summary_verbosity=0,
               distort_color_in_yiq=False,
               fuse_decode_and_crop=False):
    # Process images of this size. Depending on the model configuration, the
    # size of the input layer might differ from the original size of 32 x 32.
    self.height = height or 32
    self.width = width or 32
    self.depth = 3
    self.batch_size = batch_size
    self.num_splits = num_splits
    self.dtype = dtype
    self.train = train
    self.distortions = distortions
    self.shift_ratio = shift_ratio
    del distort_color_in_yiq
    del fuse_decode_and_crop
    del resize_method
    del shift_ratio  # unused, because a RecordInput is not used
    if self.batch_size % self.num_splits != 0:
      raise ValueError(
          ('batch_size must be a multiple of num_splits: '
           'batch_size %d, num_splits: %d') %
          (self.batch_size, self.num_splits))
    self.batch_size_per_split = self.batch_size // self.num_splits
    self.summary_verbosity = summary_verbosity

  def _distort_image(self, image):
    """Distort one image for training a network.

    Adopted the standard data augmentation scheme that is widely used for
    this dataset: the images are first zero-padded with 4 pixels on each side,
    then randomly cropped to again produce distorted images; half of the images
    are then horizontally mirrored.

    Args:
      image: input image.
    Returns:
      distored image.
    """
    image = tf.image.resize_image_with_crop_or_pad(
        image, self.height + 8, self.width + 8)
    distorted_image = tf.random_crop(image,
                                     [self.height, self.width, self.depth])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    if self.summary_verbosity >= 3:
      tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def _eval_image(self, image):
    """Get the image for model evaluation."""
    distorted_image = tf.image.resize_image_with_crop_or_pad(
        image, self.width, self.height)
    if self.summary_verbosity >= 3:
      tf.summary.image('cropped.image', tf.expand_dims(distorted_image, 0))
    return distorted_image

  def preprocess(self, raw_image):
    """Preprocessing raw image."""
    if self.summary_verbosity >= 3:
      tf.summary.image('raw.image', tf.expand_dims(raw_image, 0))
    if self.train and self.distortions:
      image = self._distort_image(raw_image)
    else:
      image = self._eval_image(raw_image)
    return image

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=-1):
    # TODO(jsimsa): Implement datasets code path
    del use_datasets, cache_data, shift_ratio
    with tf.name_scope('batch_processing'):
      all_images, all_labels = dataset.read_data_files(subset)
      all_images = tf.constant(all_images)
      all_labels = tf.constant(all_labels)
      input_image, input_label = tf.train.slice_input_producer(
          [all_images, all_labels])
      input_image = tf.cast(input_image, self.dtype)
      input_label = tf.cast(input_label, tf.int32)
      # Ensure that the random shuffling has good mixing properties.
      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(dataset.num_examples_per_epoch(subset) *
                               min_fraction_of_examples_in_queue)
      raw_images, raw_labels = tf.train.shuffle_batch(
          [input_image, input_label], batch_size=self.batch_size,
          capacity=min_queue_examples + 3 * self.batch_size,
          min_after_dequeue=min_queue_examples)

      images = [[] for i in range(self.num_splits)]
      labels = [[] for i in range(self.num_splits)]

      # Create a list of size batch_size, each containing one image of the
      # batch. Without the unstack call, raw_images[i] would still access the
      # same image via a strided_slice op, but would be slower.
      raw_images = tf.unstack(raw_images, axis=0)
      raw_labels = tf.unstack(raw_labels, axis=0)
      for i in xrange(self.batch_size):
        split_index = i % self.num_splits
        # The raw image read from data has the format [depth, height, width]
        # reshape to the format returned by minibatch.
        raw_image = tf.reshape(raw_images[i],
                               [dataset.depth, dataset.height, dataset.width])
        raw_image = tf.transpose(raw_image, [1, 2, 0])
        image = self.preprocess(raw_image)
        images[split_index].append(image)

        labels[split_index].append(raw_labels[i])

      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])
      return images, labels


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


class TestImagePreprocessor(object):
  """Preprocessor used for testing.

  set_fake_data() sets which images and labels will be output by minibatch(),
  and must be called before minibatch(). This allows tests to easily specify
  a set of images to use for training, without having to create any files.

  Queue runners must be started for this preprocessor to work.
  """

  def __init__(self,
               height,
               width,
               batch_size,
               num_splits,
               dtype,
               train=None,
               distortions=None,
               resize_method=None,
               shift_ratio=0,
               summary_verbosity=0,
               distort_color_in_yiq=False,
               fuse_decode_and_crop=False):
    del height, width, train, distortions, resize_method
    del summary_verbosity, fuse_decode_and_crop, distort_color_in_yiq
    self.batch_size = batch_size
    self.num_splits = num_splits
    self.dtype = dtype
    self.expected_subset = None
    self.shift_ratio = shift_ratio

  def set_fake_data(self, fake_images, fake_labels):
    assert len(fake_images.shape) == 4
    assert len(fake_labels.shape) == 1
    assert fake_images.shape[0] == fake_labels.shape[0]
    assert fake_images.shape[0] % self.batch_size == 0
    self.fake_images = fake_images
    self.fake_labels = fake_labels

  def minibatch(self, dataset, subset, use_datasets, cache_data,
                shift_ratio=-1):
    del dataset, use_datasets, cache_data, shift_ratio
    if (not hasattr(self, 'fake_images') or
        not hasattr(self, 'fake_labels')):
      raise ValueError('Must call set_fake_data() before calling minibatch '
                       'on TestImagePreprocessor')
    if self.expected_subset is not None:
      assert subset == self.expected_subset

    with tf.name_scope('batch_processing'):
      image_slice, label_slice = tf.train.slice_input_producer(
          [self.fake_images, self.fake_labels],
          shuffle=False,
          name='image_slice')
      raw_images, raw_labels = tf.train.batch(
          [image_slice, label_slice], batch_size=self.batch_size,
          name='image_batch')
      images = [[] for _ in range(self.num_splits)]
      labels = [[] for _ in range(self.num_splits)]
      for i in xrange(self.batch_size):
        split_index = i % self.num_splits
        raw_image = tf.cast(raw_images[i], self.dtype)
        images[split_index].append(raw_image)
        labels[split_index].append(raw_labels[i])
      for split_index in xrange(self.num_splits):
        images[split_index] = tf.parallel_stack(images[split_index])
        labels[split_index] = tf.parallel_stack(labels[split_index])

      return images, labels
