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

"""Utilities for CNN benchmarks."""
from __future__ import print_function

from collections import namedtuple
import sys
import threading

from absl import flags
import tensorflow as tf
flags.DEFINE_boolean('use_python32_barrier', False,
                     """When on, use threading.Barrier at python 3.2.""")
flags.DEFINE_boolean('flush_stdout', False,
                     """When on, flush stdout everytime log_fn is called.""")
FLAGS = flags.FLAGS

def tensorflow_version_tuple(): 
  v = tf.__version__ 
  major, minor, patch = v.split('.') 
  return (int(major), int(minor), patch) 
  
def log_fn(log):
  print(log)
  if FLAGS.flush_stdout:
    sys.stdout.flush()

class Barrier(object):
  """Implements a lightweight Barrier.

  Useful for synchronizing a fixed number of threads at known synchronization
  points.  Threads block on 'wait()' and simultaneously return once they have
  all made that call.

  # Implementaion adpoted from boost/thread/barrier.hpp
  """

  def __init__(self, parties):
    """Create a barrier, initialised to 'parties' threads."""
    self.cond = threading.Condition(threading.Lock())
    self.parties = parties
    # Indicates the number of waiting parties.
    self.waiting = 0
    # generation is needed to deal with spurious wakeups. If self.cond.wait()
    # wakes up for other reasons, generation will force it go back to wait().
    self.generation = 0
    self.broken = False

  def wait(self):
    """Wait for the barrier."""
    with self.cond:
      # Check if the barrier has been disabled or not.
      if self.broken:
        return
      gen = self.generation
      self.waiting += 1
      if self.waiting == self.parties:
        self.waiting = 0
        self.generation += 1
        self.cond.notify_all()
      # loop because of spurious wakeups
      while gen == self.generation:
        self.cond.wait()

  # TODO(huangyp): Remove this method once we find a way to know which step
  # is the last barrier.
  def abort(self):
    """Clear existing barrier and disable this barrier."""
    with self.cond:
      if self.waiting > 0:
        self.generation += 1
        self.cond.notify_all()
      self.broken = True
      
class ImageProducer(object): 

  def __init__(self, sess, put_ops, batch_group_size): 
    self.sess = sess 
    self.num_gets = 0 
    self.put_ops = put_ops 
    self.batch_group_size = batch_group_size 
    self.done_event = threading.Event() 

    if (FLAGS.use_python32_barrier and 
        sys.version_info[0] == 3 and sys.version_info[1] >= 2): 
      self.put_barrier = threading.Barrier(2) 
    else: 
      self.put_barrier = Barrier(2) 

 
  def _should_put(self): 
    return (self.num_gets + 1) % self.batch_group_size == 0 

  def done(self): 
    """Stop the image producer.""" 
    self.done_event.set() 
    self.put_barrier.abort() 
    self.thread.join() 

 

  def start(self): 
    """Start the image producer.""" 
    self.sess.run([self.put_ops]) 
    self.thread = threading.Thread(target=self._loop_producer) 
    # Set daemon to true to allow Ctrl + C to terminate all threads. 
    self.thread.daemon = True 
    self.thread.start() 

 

  def notify_image_consumption(self): 
    if self._should_put(): 
      self.put_barrier.wait() 
    self.num_gets += 1 

 

  def _loop_producer(self): 
    while not self.done_event.isSet(): 
      self.sess.run([self.put_ops]) 
      self.put_barrier.wait() 
