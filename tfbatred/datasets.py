from abc import abstractmethod
import os

import numpy as np
from six.moves import cPickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import preprocessing


IMAGENET_NUM_TRAIN_IMAGES = 1281167
IMAGENET_NUM_VAL_IMAGES = 50000


def create_dataset(data_dir, data_name):
  """Create a Dataset instance based on data_dir and data_name."""
  supported_datasets = {
      'imagenet': ImagenetData,
  }
  if not data_dir and not data_name:
    # When using synthetic data, use synthetic imagenet images by default.
    data_name = 'imagenet'

  return supported_datasets[data_name](data_dir)



      
class Dataset(object):
  """Abstract class for cnn benchmarks dataset."""

  def __init__(self, name, height=None, width=None, depth=None, data_dir=None,
               queue_runner_required=False, num_classes=1000):
    self.name = name
    self.height = height
    self.width = width
    self.depth = depth or 3

    self.data_dir = data_dir
    self._queue_runner_required = queue_runner_required
    self._num_classes = num_classes

  @property
  def num_classes(self):
    return self._num_classes

  def queue_runner_required(self):
    return self._queue_runner_required

  def use_synthetic_gpu_images(self):
    return not self.data_dir


class ImagenetData(Dataset): 
  """Configuration for Imagenet dataset.""" 
  def __init__(self, data_dir=None): 
    super(ImagenetData, self).__init__('imagenet', 300, 300, data_dir=data_dir) 

  def get_image_preprocessor(self): 
    if self.use_synthetic_gpu_images(): 
      return preprocessing.SyntheticImagePreprocessor 


