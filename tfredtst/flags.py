 
from collections import namedtuple

from absl import flags as absl_flags
import six


FLAGS = absl_flags.FLAGS


# ParamSpec describes one of benchmark_cnn.BenchmarkCNN's parameters.
ParamSpec = namedtuple('_ParamSpec',
                       ['flag_type', 'default_value', 'description',
                        'kwargs'])


# Maps from parameter name to its ParamSpec.
param_specs = {}


def DEFINE_string(name, default, help):  # pylint: disable=invalid-name,redefined-builtin
  param_specs[name] = ParamSpec('string', default, help, {})


def DEFINE_boolean(name, default, help):  # pylint: disable=invalid-name,redefined-builtin
  param_specs[name] = ParamSpec('boolean', default, help, {})


def DEFINE_integer(name, default, help, lower_bound=None, upper_bound=None):  # pylint: disable=invalid-name,redefined-builtin
  kwargs = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
  param_specs[name] = ParamSpec('integer', default, help, kwargs)


def DEFINE_float(name, default, help, lower_bound=None, upper_bound=None):  # pylint: disable=invalid-name,redefined-builtin
  kwargs = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
  param_specs[name] = ParamSpec('float', default, help, kwargs)


def DEFINE_enum(name, default, enum_values, help):  # pylint: disable=invalid-name,redefined-builtin
  kwargs = {'enum_values': enum_values}
  param_specs[name] = ParamSpec('enum', default, help, kwargs)


def DEFINE_list(name, default, help):  # pylint: disable=invalid-name,redefined-builtin
  param_specs[name] = ParamSpec('list', default, help, {})


def define_flags(specs=None):
  """Define a command line flag for each ParamSpec in flags.param_specs."""
  specs = specs or param_specs
  define_flag = {
      'boolean': absl_flags.DEFINE_boolean,
      'float': absl_flags.DEFINE_float,
      'integer': absl_flags.DEFINE_integer,
      'string': absl_flags.DEFINE_string,
      'enum': absl_flags.DEFINE_enum,
      'list': absl_flags.DEFINE_list
  }
  for name, param_spec in six.iteritems(specs):
    if param_spec.flag_type not in define_flag:
      raise ValueError('Unknown flag_type %s' % param_spec.flag_type)
    else:
      define_flag[param_spec.flag_type](name, param_spec.default_value,
                                        help=param_spec.description,
                                        **param_spec.kwargs)
