"""TensorFlow benchmark library.

See the README for more information.
"""

from __future__ import print_function

import argparse
from collections import namedtuple
import math
import multiprocessing
import os
import threading
import time

from absl import flags
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest
#import benchmark_storage
import cnn_util
import convnet_builder
import datasets
import variable_mgr
import variable_mgr_util
#from cnn_util import log_fn
#from cnn_util import ParamSpec
from models import model_config
from platforms import util as platforms_util
import params
# params._DEFAULT_PARAMS maps from each parameter's name to its ParamSpec. For each
# parameter listed here, a command line flag will be defined for
# tf_cnn_benchmarks.py.

def log_fn(log):
  print(log)



def define_flags():
  """Define a command line flag for each ParamSpec in _DEFAULT_PARAMS."""
  define_flag = {
      'boolean': flags.DEFINE_boolean,
      'float': flags.DEFINE_float,
      'integer': flags.DEFINE_integer,
      'string': flags.DEFINE_string,
  }
  for name, param_spec in six.iteritems(params._DEFAULT_PARAMS):
      define_flag[param_spec.flag_type](name, param_spec.default_value,                                       param_spec.description)
      flags.declare_key_flag(name)

FLAGS = flags.FLAGS


class GlobalStepWatcher(threading.Thread):
  """A helper class for globe_step.

  Polls for changes in the global_step of the model, and finishes when the
  number of steps for the global run are done.
  """

  def __init__(self, sess, global_step_op, start_at_global_step,
               end_at_global_step):
    threading.Thread.__init__(self)
    self.sess = sess
    self.global_step_op = global_step_op
    self.start_at_global_step = start_at_global_step
    self.end_at_global_step = end_at_global_step

    self.start_time = 0
    self.start_step = 0
    self.finish_time = 0
    self.finish_step = 0

  def run(self):
    while self.finish_time == 0:
      time.sleep(.25)
      global_step_val, = self.sess.run([self.global_step_op])
      if self.start_time == 0 and global_step_val >= self.start_at_global_step:
        # Use tf.logging.info instead of log_fn, since print (which is log_fn)
        # is not thread safe and may interleave the outputs from two parallel
        # calls to print, which can break tests.
        tf.logging.info('Starting real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.start_time = time.time()
        self.start_step = global_step_val
      if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
        tf.logging.info('Finishing real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.finish_time = time.time()
        self.finish_step = global_step_val

  def done(self):
    return self.finish_time > 0

  def num_steps(self):
    return self.finish_step - self.start_step

  def elapsed_time(self):
    return self.finish_time - self.start_time


 

def get_data_type(params):
  """Returns BenchmarkCNN's data type as determined by use_fp16.
  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """#if params.use_fp16
  if params.use_fp16 :
    return tf.float16 
  else :
    return tf.float32


def loss_function(logits, labels, aux_logits):
  """Loss function."""
  with tf.name_scope('xentropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def create_config_proto(params):
  """Returns session config proto.

  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = params.num_intra_threads
  config.inter_op_parallelism_threads = params.num_inter_threads
  config.gpu_options.force_gpu_compatible = params.force_gpu_compatible

  return config


def get_mode_from_params(params):
  return 'training'


def benchmark_one_step(sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       image_producer,
                       params,
                       summary_op=None):
  """Advance one step of benchmarking."""
   
  run_options = None
  run_metadata = None
  summary_str = None
  start_time = time.time()
   
  results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
  lossval = results['total_loss']
  
  image_producer.notify_image_consumption()
  train_time = time.time() - start_time
  step_train_times.append(train_time)
  if step >= 0 and (step == 0 or (step + 1) % params.display_every == 0):
    log_str = '%i\t%s\t%.3f' % (
        step + 1, get_perf_timing_str(batch_size, step_train_times), lossval)
    log_fn(log_str)
   
  return summary_str


def get_perf_timing_str(batch_size, step_train_times, scale=1):
  times = np.array(step_train_times)
  speeds = batch_size / times
  speed_mean = scale * batch_size / np.mean(times)
  if scale == 1:
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    speed_jitter = speed_madstd
    return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
            (speed_mean, speed_uncertainty, speed_jitter))

# Params are passed to BenchmarkCNN's constructor. Params is a map from name
# to value, with one field per key in params._DEFAULT_PARAMS.
#
# Call make_params() or make_params_from_flags() below to construct a Params
# tuple with default values from params._DEFAULT_PARAMS, rather than constructing
# Params directly.
Params = namedtuple('Params', params._DEFAULT_PARAMS.keys())  # pylint: disable=invalid-name



def make_params_from_flags():
  """Create a Params tuple for BenchmarkCNN from FLAGS.

  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Collect (name: value) pairs for FLAGS with matching names in
  # params._DEFAULT_PARAMS.
  flag_values = {name: getattr(FLAGS, name) for name in params._DEFAULT_PARAMS.keys()}
  return Params(**flag_values)


class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self, params):
    """Initialize BenchmarkCNN.

    Args:
      params: Params tuple, typically created by make_params or
              make_params_from_flags.
    Raises:
      ValueError: Unsupported params settings.
    """
    self.params = params
    self.dataset = datasets.create_dataset(self.params.data_dir,
                                           self.params.data_name)
    self.model = model_config.get_model_config(self.params.model, self.dataset)
    self.trace_filename = self.params.trace_file
    self.data_format = self.params.data_format
    self.enable_layout_optimizer = self.params.enable_layout_optimizer
    self.num_batches = self.params.num_batches
    autotune_threshold = self.params.autotune_threshold if (
        self.params.autotune_threshold) else 1
    min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
    self.num_warmup_batches = self.params.num_warmup_batches if (
        self.params.num_warmup_batches is not None) else max(
            10, min_autotune_warmup)
    self.graph_file = self.params.graph_file
    self.resize_method = self.params.resize_method
    self.sync_queue_counter = 0
    self.num_gpus = self.params.num_gpus
    
    self.gpu_indices = [x for x in range(self.num_gpus)]
    self.use_synthetic_gpu_images = self.dataset.use_synthetic_gpu_images()

    # Use the batch size from the command line if specified, otherwise use the
    # model's default batch size.  Scale the benchmark's batch size by the
    # number of GPUs.
    if self.params.batch_size > 0:
      self.model.set_batch_size(self.params.batch_size)
    self.batch_size = self.model.get_batch_size() * self.num_gpus
    self.batch_group_size = self.params.batch_group_size
    self.enable_auto_loss_scale = (
        self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale)
    self.loss_scale = None
    self.loss_scale_normal_steps = None

    self.job_name = self.params.job_name  # "" for local training

    # PS server is used for distributed jobs not using all-reduce.
    use_ps_server = self.job_name and (self.params.variable_update !=
                                       'distributed_all_reduce')
    # controller is used for distributed_all_reduce with > 1 worker.
    use_controller = (
        self.params.variable_update == 'distributed_all_reduce' and
        self.job_name)

    self.local_parameter_device_flag = self.params.local_parameter_device
    if 0 :
      aa_debug = 1
    else:
      self.task_index = 0
      self.cluster_manager = None
      worker_prefix = ''
      self.param_server_device = '/%s:0' % self.params.local_parameter_device
      self.sync_queue_devices = [self.param_server_device]

    self.num_workers = (
        self.cluster_manager.num_workers() if self.cluster_manager else 1)
    self.num_ps = self.cluster_manager.num_ps() if self.cluster_manager else 0


    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '%s/cpu:0' % worker_prefix

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = [
        '%s/%s:%i' % (worker_prefix, self.params.device, i)
        for i in xrange(self.num_gpus)
    ]

    self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    self.global_step_device = self.cpu_device

    self.image_preprocessor = self.get_image_preprocessor()
    self.init_global_step = 0


  def print_info(self):
    """Print basic information."""
    log_fn('Model:       %s' % self.model.get_model())
    dataset_name = self.dataset.name
    if self.dataset.use_synthetic_gpu_images():
      dataset_name += ' (synthetic)'
    log_fn('Dataset:     %s' % dataset_name)
    log_fn('Mode:        %s' % get_mode_from_params(self.params))
    single_session = self.params.variable_update == 'distributed_all_reduce'
    log_fn('SingleSess:  %s' % single_session)
    device_list = self.raw_devices
    batch_size = self.num_workers * self.batch_size
    log_fn('Batch size:  %s global' % batch_size)
    log_fn('             %s per device' % (batch_size / len(device_list)))
    
    log_fn('Devices:     %s' % device_list)
    log_fn('Data format: %s' % self.data_format)
    log_fn('Layout optimizer: %s' % self.enable_layout_optimizer)
    log_fn('Optimizer:   %s' % self.params.optimizer)
    log_fn('Variables:   %s' % self.params.variable_update)


  def run(self):

    with tf.Graph().as_default():
      return self._benchmark_cnn()



  def _benchmark_cnn(self):
    """Run cnn in benchmark mode. When forward_only on, it forwards CNN.

    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    self.single_session = False
    (image_producer_ops, enqueue_ops, fetches) = self._build_model()
    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list)
    execution_barrier = None
    

    global_step = tf.train.get_global_step()
    with tf.device(self.global_step_device):
      with tf.control_dependencies([main_fetch_group]):
        fetches['inc_global_step'] = global_step.assign_add(1)


    local_var_init_op = tf.local_variables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)

    summary_op = tf.summary.merge_all()
    is_chief = (not self.job_name or self.task_index == 0)
    summary_writer = None
    
    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    saver = tf.train.Saver(
        self.variable_mgr.savable_variables(), save_relative_paths=True)
    ready_for_local_init_op = None
    
    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=self.params.train_dir,
        ready_for_local_init_op=ready_for_local_init_op,
        local_init_op=local_var_init_op_group,
        saver=saver,
        global_step=global_step,
        summary_op=None,
        save_model_secs=self.params.save_model_secs,
        summary_writer=summary_writer)

    step_train_times = []
    start_standard_services = (
        self.params.summary_verbosity >= 1 or
        self.dataset.queue_runner_required())
    target = self.cluster_manager.get_target() if self.cluster_manager else ''
    with sv.managed_session(
        master=target,
        config=create_config_proto(self.params),
        start_standard_services=start_standard_services) as sess:
      image_producer = cnn_util.ImageProducer(sess, image_producer_ops,
                                              self.batch_group_size)
      image_producer.start()
      for i in xrange(len(enqueue_ops)):
        sess.run(enqueue_ops[:(i + 1)])
        image_producer.notify_image_consumption()
      self.init_global_step, = sess.run([global_step])
      if not self.single_session:
        global_step_watcher = GlobalStepWatcher(
            sess, global_step,
            self.num_workers * self.num_warmup_batches +
            self.init_global_step,
            self.num_workers * (self.num_warmup_batches + self.num_batches) - 1)
        global_step_watcher.start()
      

      log_fn('Running warm up')
      local_step = -1 * self.num_warmup_batches
      done_fn = global_step_watcher.done
      loop_start_time = time.time()
      while not done_fn():
        if local_step == 0:
          log_fn('Done warm up')
          
          header_str = 'Step\tImg/sec\tloss'
          
          log_fn(header_str)
          
          # reset times to ignore warm up batch
          step_train_times = []
          loop_start_time = time.time()
        
        fetch_summary = None
        summary_str = benchmark_one_step(
            sess, fetches, local_step,
            self.batch_size * (self.num_workers if self.single_session else 1),
            step_train_times, self.trace_filename, image_producer, self.params,
            fetch_summary)
        
        local_step += 1
      loop_end_time = time.time()
      # Waits for the global step to be done, regardless of done_fn.
      
      num_steps = global_step_watcher.num_steps()
      elapsed_time = global_step_watcher.elapsed_time()

      average_wall_time = elapsed_time / num_steps if num_steps > 0 else 0
      images_per_sec = ((self.num_workers * self.batch_size) / average_wall_time
                        if average_wall_time > 0 else 0)

      log_fn('-' * 64)
      log_fn('total images/sec: %.2f' % images_per_sec)
      log_fn('-' * 64)
      image_producer.done()
      #if is_chief:
      #  store_benchmarks({'total_images_per_sec': images_per_sec}, self.params)
      # Save the model checkpoint.
      
    sv.stop()
    return {
        'num_workers': self.num_workers,
        'num_steps': num_steps,
        'average_wall_time': average_wall_time,
        'images_per_sec': images_per_sec
    }

  def _build_image_processing(self, shift_ratio=0):
    """"Build the image (pre)processing portion of the model graph."""
    with tf.device(self.cpu_device):
      subset = 'train'
      image_producer_ops = []
      image_producer_stages = []
      images_splits, labels_splits = self.image_preprocessor.minibatch(
          self.dataset,
          subset=subset,
          use_datasets=self.params.use_datasets,
          cache_data=self.params.cache_data,
          shift_ratio=shift_ratio)
      images_shape = images_splits[0].get_shape()
      labels_shape = labels_splits[0].get_shape()
      for device_num in range(len(self.devices)):
        image_producer_stages.append(
            data_flow_ops.StagingArea(
                [images_splits[0].dtype, labels_splits[0].dtype],
                shapes=[images_shape, labels_shape]))
    return (image_producer_ops, image_producer_stages)

  def _build_model(self):
    """Build the TensorFlow graph."""
    tf.set_random_seed(self.params.tf_random_seed)
    np.random.seed(4321)
    phase_train = not (self.params.eval or self.params.forward_only)

    log_fn('Generating model')
    losses = []
    device_grads = []
    all_logits = []
    all_top_1_ops = []
    all_top_5_ops = []
    enqueue_ops = []
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()
    
    # Build the processing and model for the worker.
    (image_producer_ops,
     image_producer_stages) = self._build_image_processing(shift_ratio=0)
    image_producer_ops = tf.group(*image_producer_ops)
    update_ops = None
    staging_delta_ops = []

    for device_num in range(len(self.devices)):
      with self.variable_mgr.create_outer_variable_scope(
          device_num), tf.name_scope('tower_%i' % device_num) as name_scope:
        results = self.add_forward_pass_and_gradients(
            phase_train, device_num, device_num,
            image_producer_stages[device_num], gpu_compute_stage_ops,
            gpu_grad_stage_ops)
        if phase_train:
          losses.append(results['loss'])
          device_grads.append(results['gradvars'])
        

        if device_num == 0:
          # Retain the Batch Normalization updates operations only from the
          # first tower. These operations update the moving mean and moving
          # variance variables, which are updated (but not used) during
          # training, and used during evaluation. The moving mean and variance
          # approximate the true mean and variance across all images in the
          # dataset. Therefore, in replicated mode, these moving averages would
          # be almost identical for each tower, and so we only update and save
          # the moving averages for one tower. In parameter server mode, all
          # towers share a copy of the variables so we also only need to update
          # and save the moving averages once.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          staging_delta_ops = list(self.variable_mgr.staging_delta_ops)
    
    enqueue_ops.append(tf.group(*gpu_compute_stage_ops))

    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  enqueue_ops, update_ops, all_top_1_ops,
                                  all_top_5_ops, phase_train)
    return (image_producer_ops, enqueue_ops, fetches)

  def _build_fetches(self, global_step, all_logits, losses, device_grads,
                     enqueue_ops, update_ops, all_top_1_ops, all_top_5_ops,
                     phase_train):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {'enqueue_ops': enqueue_ops}
    
    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        total_loss = tf.reduce_mean(losses)
        avg_grads = self.variable_mgr.get_gradients_to_apply(d, gradient_state)

        gradient_clip = self.params.gradient_clip
        learning_rate = (
            self.params.learning_rate or
            self.model.get_learning_rate(global_step, self.batch_size))
        
        clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate')
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        

        loss_scale_params = variable_mgr_util.AutoLossScaleParams(
            enable_auto_loss_scale=self.enable_auto_loss_scale,
            loss_scale=self.loss_scale,
            loss_scale_normal_steps=self.loss_scale_normal_steps,
            inc_loss_scale_every_n=self.params.fp16_inc_loss_scale_every_n,
            is_chief=not self.job_name or self.task_index == 0)

        self.variable_mgr.append_apply_gradients_ops(
            gradient_state, opt, clipped_grads, training_ops, loss_scale_params)
    train_op = tf.group(*(training_ops + update_ops))

    fetches['train_op'] = train_op
    fetches['total_loss'] = total_loss
    return fetches



  def add_forward_pass_and_gradients(self, phase_train, rel_device_num,
                                     abs_device_num, image_producer_stage,
                                     gpu_compute_stage_ops, gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes + 1
    input_data_type = get_data_type(self.params)
    data_type = get_data_type(self.params)
    
    with tf.device(self.raw_devices[rel_device_num]):
      if 0:
        aa_debug = 1
      else:
        # Minor hack to avoid H2D copy when using synthetic data
        image_size = self.model.get_image_size()
        image_shape = [
            self.batch_size // self.num_gpus, image_size, image_size,
            self.dataset.depth
        ]
        labels_shape = [self.batch_size // self.num_gpus]
        # Synthetic image should be within [0, 255].
        images = tf.truncated_normal(
            image_shape,
            dtype=input_data_type,
            mean=127,
            stddev=60,
            name='synthetic_images')
        images = tf.contrib.framework.local_variable(
            images, name='gpu_cached_images')
        labels = tf.random_uniform(
            labels_shape,
            minval=0,
            maxval=nclass - 1,
            dtype=tf.int32,
            name='synthetic_labels')

    with tf.device(self.devices[rel_device_num]):
      # Rescale from [0, 255] to [0, 2]
      images = tf.multiply(images, 1. / 127.5)
      # Rescale to [-1, 1]
      images = tf.subtract(images, 1.0)

      if self.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
      var_type = tf.float32
      network = convnet_builder.ConvNetBuilder(
          images, self.dataset.depth, phase_train, self.params.use_tf_layers,
          self.data_format, data_type, var_type)
      with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
        self.model.add_inference(network)
        # Add the final fully-connected class layer
        logits = network.affine(nclass, activation='linear')
        aux_logits = None

      results = {}  # The return value
      
      loss = loss_function(logits, labels, aux_logits=aux_logits)
      params = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num)
      l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
      weight_decay = self.params.weight_decay
      if weight_decay is not None and weight_decay != 0.:
        loss += weight_decay * l2_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      scaled_loss = loss if self.loss_scale is None else loss * self.loss_scale
      grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
      
      param_refs = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num, writable=True)
      gradvars = list(zip(grads, param_refs))
      results['loss'] = loss
      results['gradvars'] = gradvars
      return results

  def get_image_preprocessor(self):
    """Returns the image preprocessor to used, based on the model.

    Returns:
      The image preprocessor, or None if synthetic data should be used.
    """
    image_size = self.model.get_image_size()
    input_data_type = get_data_type(self.params)

    shift_ratio = 0

    processor_class = self.dataset.get_image_preprocessor()
    assert processor_class
    return processor_class(
        image_size,
        image_size,
        self.batch_size * self.batch_group_size,
        len(self.devices) * self.batch_group_size,
        dtype=input_data_type,
        train=(not self.params.eval),
        distortions=self.params.distortions,
        resize_method=self.resize_method,
        shift_ratio=shift_ratio,
        summary_verbosity=self.params.summary_verbosity,
        distort_color_in_yiq=self.params.distort_color_in_yiq,
        fuse_decode_and_crop=self.params.fuse_decode_and_crop)



def setup(params):
  """Sets up the environment that BenchmarkCNN should run in.
  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  Returns:
    A potentially modified params.
  Raises:
    ValueError: invalid parames combinations.
  """
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  os.environ['TF_SYNC_ON_FINISH'] = str(int(params.sync_on_finish))
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)


  # Sets GPU thread settings
  params = params._replace(gpu_thread_mode=params.gpu_thread_mode.lower())
  os.environ['TF_GPU_THREAD_MODE'] = params.gpu_thread_mode

  # Default to two threads. One for the device compute and the other for
  # memory copies.
  per_gpu_thread_count = params.per_gpu_thread_count or 2
  total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)

  if not params.num_inter_threads and params.gpu_thread_mode in [
      'gpu_private', 'gpu_shared'
  ]:
    cpu_count = multiprocessing.cpu_count()
    main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
    params = params._replace(num_inter_threads=main_thread_count)

  platforms_util.initialize(params, create_config_proto(params))

  return params

