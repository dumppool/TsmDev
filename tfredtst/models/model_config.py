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

"""Model configurations for CNN benchmarks.
"""
from models import resnet_model
#from models import alexnet_model
#from models import densenet_model
#from models import googlenet_model
#from models import inception_model
#from models import lenet_model
#from models import overfeat_model
#from models import resnet_model
#from models import trivial_model
#from models import vgg_model


_model_name_to_imagenet_model = {
#    'vgg11': vgg_model.Vgg11Model,
#    'vgg16': vgg_model.Vgg16Model,
#    'vgg19': vgg_model.Vgg19Model,
#    'lenet': lenet_model.Lenet5Model,
#    'googlenet': googlenet_model.GooglenetModel,
#    'overfeat': overfeat_model.OverfeatModel,
#    'alexnet': alexnet_model.AlexnetModel,
#    'trivial': trivial_model.TrivialModel,
#    'inception3': inception_model.Inceptionv3Model,
#    'inception4': inception_model.Inceptionv4Model,
    'resnet50': resnet_model.create_resnet50_model,
    'resnet50_v2': resnet_model.create_resnet50_v2_model,
    'resnet101': resnet_model.create_resnet101_model,
    'resnet101_v2': resnet_model.create_resnet101_v2_model,
    'resnet152': resnet_model.create_resnet152_model,
    'resnet152_v2': resnet_model.create_resnet152_v2_model,
}


_model_name_to_cifar_model = {
#    'alexnet': alexnet_model.AlexnetCifar10Model,
    'resnet20': resnet_model.create_resnet20_cifar_model,
    'resnet20_v2': resnet_model.create_resnet20_v2_cifar_model,
    'resnet32': resnet_model.create_resnet32_cifar_model,
    'resnet32_v2': resnet_model.create_resnet32_v2_cifar_model,
    'resnet44': resnet_model.create_resnet44_cifar_model,
    'resnet44_v2': resnet_model.create_resnet44_v2_cifar_model,
    'resnet56': resnet_model.create_resnet56_cifar_model,
    'resnet56_v2': resnet_model.create_resnet56_v2_cifar_model,
    'resnet110': resnet_model.create_resnet110_cifar_model,
    'resnet110_v2': resnet_model.create_resnet110_v2_cifar_model,
#    'trivial': trivial_model.TrivialCifar10Model,
#    'densenet40_k12': densenet_model.create_densenet40_k12_model,
#    'densenet100_k12': densenet_model.create_densenet100_k12_model,
#    'densenet100_k24': densenet_model.create_densenet100_k24_model,
}

def _get_model_map(dataset_name): 
    return _model_name_to_imagenet_model 
    
def get_model_config(model_name, dataset): 
  """Map model name to model network configuration.""" 
  model_map = _get_model_map(dataset.name) 
  return model_map[model_name]() 