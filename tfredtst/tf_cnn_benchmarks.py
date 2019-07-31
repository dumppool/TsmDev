"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function

from absl import flags
import tensorflow as tf

import benchmark_cnn
import cnn_util
from cnn_util import log_fn
import proparams

benchmark_cnn.define_flags()
flags.adopt_module_key_flags(benchmark_cnn)
 

def main(_):
  #params = benchmark_cnn.make_params_from_flags()
  params = benchmark_cnn.Params(cache_data=False, fp16_vars=False, graph_file=None, num_batches=100, kmp_blocktime=30, eval_dir='/tmp/tf_cnn_benchmarks/eval', forward_only=False, cross_replica_sync=True, gradient_clip=None, distort_color_in_yiq=False, winograd_nonfused=True, ps_hosts='', eval_interval_secs=0, controller_host=None, num_inter_threads=0, force_gpu_compatible=True, use_fp16=False, result_storage=None, batch_size=32, batch_group_size=1, fuse_decode_and_crop=True, data_name=None, enable_layout_optimizer=False, variable_update='parameter_server', autotune_threshold=None, use_datasets=True, staged_vars=False, mkl=False, model='resnet50', num_gpus=2, save_model_secs=0, agg_small_grads_max_group=10, rmsprop_momentum=0.9, job_name='', xla=False, worker_hosts='', minimum_learning_rate=0.0, resize_method='bilinear', trace_file='', learning_rate=None, num_intra_threads=1, momentum=0.9, kmp_settings=1, fp16_loss_scale=None, gpu_thread_mode='gpu_private', save_summaries_steps=0, all_reduce_spec=None, num_warmup_batches=None, train_dir=None, summary_verbosity=0, server_protocol='grpc', optimizer='sgd', per_gpu_thread_count=0, agg_small_grads_max_bytes=0, tf_random_seed=1234, num_epochs_per_decay=0.0, kmp_affinity='granularity=fine,verbose,compact,1,0', gpu_indices='', display_every=10, rmsprop_decay=0.9, weight_decay=4e-05, use_tf_layers=True, fp16_enable_auto_loss_scale=False, fp16_inc_loss_scale_every_n=1000, device='gpu', sync_on_finish=False, data_format='NCHW', eval=False, rmsprop_epsilon=1.0, task_index=0, learning_rate_decay_factor=0.0, data_dir=None, gpu_memory_frac_for_testing=0.0, print_training_accuracy=False, local_parameter_device='gpu', distortions=True)
  #print(params)
  params = benchmark_cnn.setup(params)
  bench = benchmark_cnn.BenchmarkCNN(params)

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()


if __name__ == '__main__':
  tf.app.run()
