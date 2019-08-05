

from __future__ import print_function

import tensorflow as tf

#import allreduce
import variable_mgr_util


class VariableMgr(object):
  """Abstract superclass for class used by BenchmarkCnn to control variables.

    Functions on this class are used to control how variables are created and
    managed, and how gradients are computed and applied.
  """
  def __init__(self, benchmark_cnn):
    self.benchmark_cnn = benchmark_cnn
    self.staging_delta_ops = []

    # A variable for automatic loss scaling.
    self.grad_has_inf_nan = None


  def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops,
                                 loss_scale_params):
    """Adds training ops for grads to 'training_ops'.



    Args:
      gradient_state: from previous call to apply_gradients_devices.
      opt: the underlying optimizer
      grads: [(grad, var)] to apply
      training_ops: list to which to add ops
      loss_scale_params: parameters for loss scaling.
    """
    del gradient_state  # unused by this implementation

    def get_apply_gradients_ops_func():
      """Returns the apply_gradients op."""
      return [opt.apply_gradients(grads)]

    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops, get_apply_gradients_ops_func, loss_scale_params,
        self.grad_has_inf_nan)

  def get_post_init_ops(self):
    """Returns ops that should run post-initialization."""
    return []


  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    return tf.global_variables()

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
                                    writable=False):
    """Return the set of trainable variables on device.

    Args:
      rel_device_num: local worker device index.
      abs_device_num: global graph device index.
      writable: whether to get a reference to the underlying variable.

    Returns:
      The set of trainable vairalbes on the specified device.
    """
    del rel_device_num, writable
    params = tf.trainable_variables()
    return params



class VariableMgrLocalFetchFromPS(VariableMgr):
  """VariableMgr that implements the --parameter_server mode for local jobs.

     Variables are stored on a parameter server.  For each step, each tower gets
     a copy of the variables from the parameter server, and sends its gradients
     to the param server.
  """

  def each_tower_has_variables(self):
    return False

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v', reuse=bool(device_num))

  def preprocess_device_grads(self, device_grads):
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    assert device_num == 0
    device_grads = gradient_state
    agg_grads, self.grad_has_inf_nan = (
        variable_mgr_util.
        aggregate_gradients_using_copy_with_variable_colocation(
            device_grads,
            use_mean=True,
            check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))
    return agg_grads

  def get_devices(self):
    raw_devices = self.benchmark_cnn.raw_devices
    if self.benchmark_cnn.local_parameter_device_flag == 'gpu':
      return [
          variable_mgr_util.ParamServerDeviceSetter(d, raw_devices)
          for d in raw_devices
      ]
