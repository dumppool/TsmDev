"""CNN builder."""

from __future__ import print_function

from collections import defaultdict
import contextlib

import numpy as np

import tensorflow as tf

from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.training import moving_averages


class ConvNetBuilder(object):
  """Builder of cnn net."""

  def __init__(self,
               input_op,
               input_nchan,
               phase_train,
               use_tf_layers,
               data_format='NCHW',
               dtype=tf.float32,
               variable_dtype=tf.float32):
    self.top_layer = input_op
    self.top_size = input_nchan
    self.phase_train = phase_train
    self.use_tf_layers = use_tf_layers
    self.data_format = data_format
    self.dtype = dtype
    self.variable_dtype = variable_dtype
    self.counts = defaultdict(lambda: 0)
    self.use_batch_norm = False
    self.batch_norm_config = {}  # 'decay': 0.997, 'scale': True}
    self.channel_pos = ('channels_last'
                        if data_format == 'NHWC' else 'channels_first')
    self.aux_top_layer = None
    self.aux_top_size = 0

  def get_custom_getter(self):
    """Returns a custom getter that this class's methods must be called under.

    All methods of this class must be called under a variable scope that was
    passed this custom getter. Example:

    ```python
    network = ConvNetBuilder(...)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      network.conv(...)
      # Call more methods of network here
    ```

    Currently, this custom getter only does anything if self.use_tf_layers is
    True. In that case, it causes variables to be stored as dtype
    self.variable_type, then casted to the requested dtype, instead of directly
    storing the variable as the requested dtype.
    """
    def inner_custom_getter(getter, *args, **kwargs):
      """Custom getter that forces variables to have type self.variable_type."""
      requested_dtype = kwargs['dtype']
      if not (requested_dtype == tf.float32 and
              self.variable_dtype == tf.float16):
        # Only change the variable dtype if doing so does not decrease variable
        # precision.
        kwargs['dtype'] = self.variable_dtype
      var = getter(*args, **kwargs)
      # This if statement is needed to guard the cast, because batch norm
      # assigns directly to the return value of this custom getter. The cast
      # makes the return value not a variable so it cannot be assigned. Batch
      # norm variables are always in fp32 so this if statement is never
      # triggered for them.
      return var
    return inner_custom_getter


  def get_variable(self, name, shape, dtype, cast_dtype, *args, **kwargs):
    # TODO(reedwm): Currently variables and gradients are transferred to other
    # devices and machines as type `dtype`, not `cast_dtype`. In particular,
    # this means in fp16 mode, variables are transferred as fp32 values, not
    # fp16 values, which uses extra bandwidth.
    var = tf.get_variable(name, shape, dtype, *args, **kwargs)
    return tf.cast(var, cast_dtype)

  def _conv2d_impl(self, input_layer, num_channels_in, filters, kernel_size,
                   strides, padding, kernel_initializer):
    return conv_layers.conv2d(input_layer, filters, kernel_size, strides,
                                padding, self.channel_pos,
                                kernel_initializer=kernel_initializer,
                                use_bias=False)
     

  def conv(self,
           num_out_channels,
           k_height,
           k_width,
           d_height=1,
           d_width=1,
           mode='SAME',
           input_layer=None,
           num_channels_in=None,
           use_batch_norm=None,
           stddev=None,
           activation='relu',
           bias=0.0):
    """Construct a conv2d layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    kernel_initializer = None
    if stddev is not None:
      kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    name = 'conv' + str(self.counts['conv'])
    self.counts['conv'] += 1
    with tf.variable_scope(name):
      strides = [1, d_height, d_width, 1]
      if self.data_format == 'NCHW':
        strides = [strides[0], strides[3], strides[1], strides[2]]
      if mode != 'SAME_RESNET':
        conv = self._conv2d_impl(input_layer, num_channels_in, num_out_channels,
                                 kernel_size=[k_height, k_width],
                                 strides=[d_height, d_width], padding=mode,
                                 kernel_initializer=kernel_initializer)
      else:  # Special padding mode for ResNet models
        if d_height == 1 and d_width == 1:
          conv = self._conv2d_impl(input_layer, num_channels_in,
                                   num_out_channels,
                                   kernel_size=[k_height, k_width],
                                   strides=[d_height, d_width], padding='SAME',
                                   kernel_initializer=kernel_initializer)
        else:
          rate = 1  # Unused (for 'a trous' convolutions)
          kernel_height_effective = k_height + (k_height - 1) * (rate - 1)
          pad_h_beg = (kernel_height_effective - 1) // 2
          pad_h_end = kernel_height_effective - 1 - pad_h_beg
          kernel_width_effective = k_width + (k_width - 1) * (rate - 1)
          pad_w_beg = (kernel_width_effective - 1) // 2
          pad_w_end = kernel_width_effective - 1 - pad_w_beg
          padding = [[0, 0], [pad_h_beg, pad_h_end],
                     [pad_w_beg, pad_w_end], [0, 0]]
          if self.data_format == 'NCHW':
            padding = [padding[0], padding[3], padding[1], padding[2]]
          input_layer = tf.pad(input_layer, padding)
          conv = self._conv2d_impl(input_layer, num_channels_in,
                                   num_out_channels,
                                   kernel_size=[k_height, k_width],
                                   strides=[d_height, d_width], padding='VALID',
                                   kernel_initializer=kernel_initializer)
      if use_batch_norm is None:
        use_batch_norm = self.use_batch_norm
      if not use_batch_norm:
        if bias is not None:
          biases = self.get_variable('biases', [num_out_channels],
                                     self.variable_dtype, self.dtype,
                                     initializer=tf.constant_initializer(bias))
          biased = tf.reshape(
              tf.nn.bias_add(conv, biases, data_format=self.data_format),
              conv.get_shape())
        else:
          biased = conv
      else:
        self.top_layer = conv
        self.top_size = num_out_channels
        biased = self.batch_norm(**self.batch_norm_config)
      if activation == 'relu':
        conv1 = tf.nn.relu(biased)
      elif activation == 'linear' or activation is None:
        conv1 = biased
      elif activation == 'tanh':
        conv1 = tf.nn.tanh(biased)
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = conv1
      self.top_size = num_out_channels
      return conv1

  def _pool(self,
            pool_name,
            pool_function,
            k_height,
            k_width,
            d_height,
            d_width,
            mode,
            input_layer,
            num_channels_in):
    """Construct a pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = pool_name + str(self.counts[pool_name])
    self.counts[pool_name] += 1
    if self.use_tf_layers:
      pool = pool_function(
          input_layer, [k_height, k_width], [d_height, d_width],
          padding=mode,
          data_format=self.channel_pos,
          name=name)
    self.top_layer = pool
    return pool

  def mpool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct a max pooling layer."""
    return self._pool('mpool', pooling_layers.max_pooling2d, k_height, k_width,
                      d_height, d_width, mode, input_layer, num_channels_in)



  def affine(self,
             num_out_channels,
             input_layer=None,
             num_channels_in=None,
             bias=0.0,
             stddev=None,
             activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'affine' + str(self.counts['affine'])
    self.counts['affine'] += 1
    with tf.variable_scope(name):
      init_factor = 2. if activation == 'relu' else 1.
      stddev = stddev or np.sqrt(init_factor / num_channels_in)
      kernel = self.get_variable(
          'weights', [num_channels_in, num_out_channels],
          self.variable_dtype, self.dtype,
          initializer=tf.truncated_normal_initializer(stddev=stddev))
      biases = self.get_variable('biases', [num_out_channels],
                                 self.variable_dtype, self.dtype,
                                 initializer=tf.constant_initializer(bias))
      logits = tf.nn.xw_plus_b(input_layer, kernel, biases)
      if activation == 'relu':
        affine1 = tf.nn.relu(logits, name=name)
      elif activation == 'linear' or activation is None:
        affine1 = logits
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = affine1
      self.top_size = num_out_channels
      return affine1


  def spatial_mean(self, keep_dims=False):
    name = 'spatial_mean' + str(self.counts['spatial_mean'])
    self.counts['spatial_mean'] += 1
    axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
    self.top_layer = tf.reduce_mean(
        self.top_layer, axes, keep_dims=keep_dims, name=name)
    return self.top_layer
   
   
  def batch_norm(self, input_layer=None, decay=0.999, scale=False,
                 epsilon=0.001):
    """Adds a Batch Normalization layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'batchnorm' + str(self.counts['batchnorm'])
    self.counts['batchnorm'] += 1

    with tf.variable_scope(name) as scope:
      if self.use_tf_layers:
        bn = tf.contrib.layers.batch_norm(
            input_layer,
            decay=decay,
            scale=scale,
            epsilon=epsilon,
            is_training=self.phase_train,
            fused=True,
            data_format=self.data_format,
            scope=scope)
      else:
        bn = self._batch_norm_without_layers(input_layer, decay, scale, epsilon)
    self.top_layer = bn
    self.top_size = bn.shape[3] if self.data_format == 'NHWC' else bn.shape[1]
    self.top_size = int(self.top_size)
    return bn


