from collections import namedtuple
from platforms import util as platforms_util

ParamSpec = namedtuple('_ParamSpec',['flag_type', 'default_value', 'description']) 

_DEFAULT_PARAMS = {
    'model':
        ParamSpec('string', 'trivial', 'name of the model to run'),

    # The code will first check if it's running under benchmarking mode
    # or evaluation mode, depending on 'eval':
    # Under the evaluation mode, this script will read a saved model,
    #   and compute the accuracy of the model against a validation dataset.
    #   Additional ops for accuracy and top_k predictors are only used under
    #   this mode.
    # Under the benchmarking mode, user can specify whether nor not to use
    #   the forward-only option, which will only compute the loss function.
    #   forward-only cannot be enabled with eval at the same time.
    'eval':
        ParamSpec('boolean', False, 'whether use eval or benchmarking'),
    'eval_interval_secs':
        ParamSpec('integer', 0,
                  'How often to run eval on saved checkpoints. Usually the '
                  'same as save_model_secs from the corresponding training '
                  'run. Pass 0 to eval only once.'),
    'forward_only':
        ParamSpec('boolean', False,
                  'whether use forward-only or training for benchmarking'),
    'print_training_accuracy':
        ParamSpec('boolean', False,
                  'whether to calculate and print training accuracy during '
                  'training'),
    'batch_size':
        ParamSpec('integer', 0, 'batch size per compute device'),
    'batch_group_size':
        ParamSpec('integer', 1,
                  'number of groups of batches processed in the image '
                  'producer.'),
    'num_batches':
        ParamSpec('integer', 100, 'number of batches to run, excluding '
                  'warmup'),
    'num_warmup_batches':
        ParamSpec('integer', None, 'number of batches to run before timing'),
    'autotune_threshold':
        ParamSpec('integer', None, 'The autotune threshold for the models'),
    'num_gpus':
        ParamSpec('integer', 1, 'the number of GPUs to run on'),
    'gpu_indices':
        ParamSpec('string', '', 'indices of worker GPUs in ring order'),
    'display_every':
        ParamSpec('integer', 10,
                  'Number of local steps after which progress is printed out'),
    'data_dir':
        ParamSpec('string', None,
                  'Path to dataset in TFRecord format (aka Example '
                  'protobufs). If not specified, synthetic data will be '
                  'used.'),
    'data_name':
        ParamSpec('string', None,
                  'Name of dataset: imagenet or cifar10. If not specified, it '
                  'is automatically guessed based on data_dir.'),
    'resize_method':
        ParamSpec('string', 'bilinear',
                  'Method for resizing input images: crop, nearest, bilinear, '
                  'bicubic, area, or round_robin. The `crop` mode requires '
                  'source images to be at least as large as the network input '
                  'size. The `round_robin` mode applies different resize '
                  'methods based on position in a batch in a round-robin '
                  'fashion. Other modes support any sizes and apply random '
                  'bbox distortions before resizing (even with '
                  'distortions=False).'),
    'distortions':
        ParamSpec('boolean', True,
                  'Enable/disable distortions during image preprocessing. '
                  'These include bbox and color distortions.'),
    'use_datasets':
        ParamSpec('boolean', True, 'Enable use of datasets for input pipeline'),
    'gpu_thread_mode':
        ParamSpec('string', 'gpu_private',
                  'Methods to assign GPU host work to threads. '
                  'global: all GPUs and CPUs share the same global threads; '
                  'gpu_private: a private threadpool for each GPU; '
                  'gpu_shared: all GPUs share the same threadpool.'),
    'per_gpu_thread_count':
        ParamSpec('integer', 0, 'The number of threads to use for GPU.'
                  'Only valid when gpu_thread_mode is not global.'),
    'cache_data':
        ParamSpec('boolean', False,
                  'Enable use of a special datasets pipeline that reads a '
                  'single TFRecord into memory and repeats it infinitely many '
                  'times. The purpose of this flag is to make it possible '
                  'to write regression tests that are not bottlenecked by CNS '
                  'throughput.'),
    'local_parameter_device':
        ParamSpec('string', 'gpu',
                  'Device to use as parameter server: cpu or gpu. For '
                  'distributed training, it can affect where caching of '
                  'variables happens.'),
    'device':
        ParamSpec('string', 'gpu', 'Device to use for computation: cpu or gpu'),
    'data_format':
        ParamSpec('string', 'NCHW',
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).'),
    'num_intra_threads':
        ParamSpec('integer', 1,
                  'Number of threads to use for intra-op parallelism. If set '
                  'to 0, the system will pick an appropriate number.'),
    'num_inter_threads':
        ParamSpec('integer', 0,
                  'Number of threads to use for inter-op parallelism. If set '
                  'to 0, the system will pick an appropriate number.'),
    'trace_file':
        ParamSpec('string', '',
                  'Enable TensorFlow tracing and write trace to this file.'),
    'graph_file':
        ParamSpec('string', None,
                  'Write the model\'s graph definition to this file. Defaults '
                  'to binary format unless filename ends in `txt`.'),
    'optimizer':
        ParamSpec('string', 'sgd',
                  'Optimizer to use: momentum or sgd or rmsprop'),
    'learning_rate':
        ParamSpec('float', None, 'Initial learning rate for training.'),
    'num_epochs_per_decay':
        ParamSpec('float', 0,
                  'Steps after which learning rate decays. If 0, the learning '
                  'rate does not decay.'),
    'learning_rate_decay_factor':
        ParamSpec('float', 0,
                  'Learning rate decay factor. Decay by this factor every '
                  '`num_epochs_per_decay` epochs. If 0, learning rate does '
                  'not decay.'),
    'minimum_learning_rate':
        ParamSpec('float', 0,
                  'The minimum learning rate. The learning rate will '
                  'never decay past this value. Requires `learning_rate`, '
                  '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
                  'be set.'),
    'momentum':
        ParamSpec('float', 0.9, 'Momentum for training.'),
    'rmsprop_decay':
        ParamSpec('float', 0.9, 'Decay term for RMSProp.'),
    'rmsprop_momentum':
        ParamSpec('float', 0.9, 'Momentum in RMSProp.'),
    'rmsprop_epsilon':
        ParamSpec('float', 1.0, 'Epsilon term for RMSProp.'),
    'gradient_clip':
        ParamSpec('float', None,
                  'Gradient clipping magnitude. Disabled by default.'),
    'weight_decay':
        ParamSpec('float', 0.00004, 'Weight decay factor for training.'),
    'gpu_memory_frac_for_testing':
        ParamSpec('float', 0,
                  'If non-zero, the fraction of GPU memory that will be used. '
                  'Useful for testing the benchmark script, as this allows '
                  'distributed mode to be run on a single machine. For '
                  'example, if there are two tasks, each can be allocated '
                  '~40 percent of the memory on a single machine'),
    'use_tf_layers':
        ParamSpec('boolean', True,
                  'If True, use tf.layers for neural network layers. This '
                  'should not affect performance or accuracy in any way.'),
    'tf_random_seed':
        ParamSpec('integer', 1234,
                  'The TensorFlow random seed. Useful for debugging NaNs, as '
                  'this can be set to various values to see if the NaNs '
                  'depend on the seed.'),

    # Performance tuning parameters.
    'winograd_nonfused':
        ParamSpec('boolean', True,
                  'Enable/disable using the Winograd non-fused algorithms.'),
    'sync_on_finish':
        ParamSpec('boolean', False,
                  'Enable/disable whether the devices are synced after each '
                  'step.'),
    'staged_vars':
        ParamSpec('boolean', False,
                  'whether the variables are staged from the main '
                  'computation'),
    'force_gpu_compatible':
        ParamSpec('boolean', True,
                  'whether to enable force_gpu_compatible in GPU_Options'),
    'xla':
        ParamSpec('boolean', False, 'whether to enable XLA'),
    'fuse_decode_and_crop':
        ParamSpec('boolean', True,
                  'Fuse decode_and_crop for image preprocessing.'),
    'distort_color_in_yiq':
        ParamSpec('boolean', False,
                  'Distort color of input images in YIQ space.'),
    'enable_layout_optimizer':
        ParamSpec('boolean', False, 'whether to enable layout optimizer'),
    # Performance tuning specific to MKL.
    'mkl':
        ParamSpec('boolean', False, 'If true, set MKL environment variables.'),
    'kmp_blocktime':
        ParamSpec('integer', 30,
                  'The time, in milliseconds, that a thread should wait, '
                  'after completing the execution of a parallel region, '
                  'before sleeping'),
    'kmp_affinity':
        ParamSpec('string', 'granularity=fine,verbose,compact,1,0',
                  'Restricts execution of certain threads (virtual execution '
                  'units) to a subset of the physical processing units in a '
                  'multiprocessor computer.'),
    'kmp_settings':
        ParamSpec('integer', 1, 'If set to 1, MKL settings will be printed.'),

    # fp16 parameters. If use_fp16=False, no other fp16 parameters apply.
    'use_fp16':
        ParamSpec('boolean', False,
                  'Use 16-bit floats for certain tensors instead of 32-bit '
                  'floats. This is currently experimental.'),
    # TODO(reedwm): The default loss scale of 128 causes most models to diverge
    # on the second step with synthetic data. Changing the tf.set_random_seed
    # call to tf.set_random_seed(1235) or most other seed values causes the
    # issue not to occur.
    'fp16_loss_scale':
        ParamSpec('float', None,
                  'If fp16 is enabled, the loss is multiplied by this amount '
                  'right before gradients are computed, then each gradient '
                  'is divided by this amount. Mathematically, this has no '
                  'effect, but it helps avoid fp16 underflow. Set to 1 to '
                  'effectively disable.'),
    'fp16_vars':
        ParamSpec('boolean', False,
                  'If fp16 is enabled, also use fp16 for variables. If False, '
                  'the variables are stored in fp32 and casted to fp16 when '
                  'retrieved.  Recommended to leave as False.'),
    'fp16_enable_auto_loss_scale':
        ParamSpec('boolean', False,
                  'If True and use_fp16 is True, automatically adjust the loss'
                  ' scale during training.'),
    'fp16_inc_loss_scale_every_n':
        ParamSpec('integer', 1000,
                  'If fp16 is enabled and fp16_enable_auto_loss_scale is True,'
                  ' increase the loss scale every n steps.'),

    # The method for managing variables:
    #   parameter_server: variables are stored on a parameter server that holds
    #       the master copy of the variable. In local execution, a local device
    #       acts as the parameter server for each variable; in distributed
    #       execution, the parameter servers are separate processes in the
    #       cluster.
    #       For each step, each tower gets a copy of the variables from the
    #       parameter server, and sends its gradients to the param server.
    #   replicated: each GPU has its own copy of the variables. To apply
    #       gradients, an all_reduce algorithm or or regular cross-device
    #       aggregation is used to replicate the combined gradients to all
    #       towers (depending on all_reduce_spec parameter setting).
    #   independent: each GPU has its own copy of the variables, and gradients
    #       are not shared between towers. This can be used to check performance
    #       when no data is moved between GPUs.
    #   distributed_replicated: Distributed training only. Each GPU has a copy
    #       of the variables, and updates its copy after the parameter servers
    #       are all updated with the gradients from all servers. Only works with
    #       cross_replica_sync=true. Unlike 'replicated', currently never uses
    #       nccl all-reduce for replicating within a server.
    #   distributed_all_reduce: Distributed training where all replicas run
    #       in a single session, using all-reduce to mutally reduce the
    #       gradients.  Uses no parameter servers.  When there is only one
    #       worker, this is the same as replicated.
    'variable_update':
        ParamSpec('string', 'parameter_server',
                  'The method for managing variables: parameter_server, '
                  'replicated, distributed_replicated, independent, '
                  'distributed_all_reduce'),
    'all_reduce_spec':
        ParamSpec('string', None,
                  'A specification of the all_reduce algorithm to be used for '
                  'reducing gradients.  For more details, see '
                  'parse_all_reduce_spec in variable_mgr.py.  An '
                  'all_reduce_spec has BNF form:\n'
                  'int ::= positive whole number\n'
                  'g_int ::= int[KkMGT]?\n'
                  'alg_spec ::= alg | alg#int\n'
                  'range_spec ::= alg_spec | alg_spec/alg_spec\n'
                  'spec ::= range_spec | range_spec:g_int:range_spec\n'
                  'NOTE: not all syntactically correct constructs are '
                  'supported.\n\n'
                  'Examples:\n '
                  '"xring" == use one global ring reduction for all '
                  'tensors\n'
                  '"pscpu" == use CPU at worker 0 to reduce all tensors\n'
                  '"nccl" == use NCCL to locally reduce all tensors.  '
                  'Limited to 1 worker.\n'
                  '"nccl/xring" == locally (to one worker) reduce values '
                  'using NCCL then ring reduce across workers.\n'
                  '"pscpu:32k:xring" == use pscpu algorithm for tensors of '
                  'size up to 32kB, then xring for larger tensors.'),

    # If variable_update==distributed_all_reduce then it may be advantageous
    # to aggregate small tensors into one prior to reduction.  These parameters
    # control that aggregation.
    'agg_small_grads_max_bytes':
        ParamSpec('integer', 0, 'If > 0, try to aggregate tensors of less '
                  'than this number of bytes prior to all-reduce.'),
    'agg_small_grads_max_group':
        ParamSpec('integer', 10, 'When aggregating small tensors for '
                  'all-reduce do not aggregate more than this many into one '
                  'new tensor.'),

    # Distributed training parameters.
    'job_name':
        ParamSpec('string', '',
                  'One of "ps", "worker", "".  Empty for local training'),
    'ps_hosts':
        ParamSpec('string', '', 'Comma-separated list of target hosts'),
    'worker_hosts':
        ParamSpec('string', '', 'Comma-separated list of target hosts'),
    'controller_host':
        ParamSpec('string', None, 'optional controller host'),
    'task_index':
        ParamSpec('integer', 0, 'Index of task within the job'),
    'server_protocol':
        ParamSpec('string', 'grpc', 'protocol for servers'),
    'cross_replica_sync':
        ParamSpec('boolean', True, ''),

    # Summary and Save & load checkpoints.
    'summary_verbosity':
        ParamSpec(
            'integer', 0, 'Verbosity level for summary ops. '
            '  level 0: disable any summary. '
            '  level 1: small and fast ops, e.g.: learning_rate, total_loss.'
            '  level 2: medium-cost ops, e.g. histogram of all gradients.'
            '  level 3: expensive ops: images and histogram of each gradient.'),
    'save_summaries_steps':
        ParamSpec('integer', 0,
                  'How often to save summaries for trained models. Pass 0 to '
                  'disable summaries.'),
    'save_model_secs':
        ParamSpec('integer', 0,
                  'How often to save trained models. Pass 0 to disable '
                  'checkpoints.'),
    'train_dir':
        ParamSpec('string', None,
                  'Path to session checkpoints. Pass None to disable saving '
                  'checkpoint at the end.'),
    'eval_dir':
        ParamSpec('string', '/tmp/tf_cnn_benchmarks/eval',
                  'Directory where to write eval event logs.'),
    'result_storage':
        ParamSpec('string', None,
                  'Specifies storage option for benchmark results. None means '
                  'results won\'t be stored. `cbuild_benchmark_datastore` '
                  'means results will be stored in cbuild datastore (note: '
                  'this option requires special permissions and meant to be '
                  'used from cbuilds).'),
}
_DEFAULT_PARAMS.update(platforms_util.get_platform_params())
