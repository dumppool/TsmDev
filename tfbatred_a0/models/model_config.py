
from models import resnet_model


_model_name_to_imagenet_model = {
    'resnet50': resnet_model.create_resnet50_model,
   # 'resnet50_v2': resnet_model.create_resnet50_v2_model,
   # 'resnet101': resnet_model.create_resnet101_model,
   # 'resnet101_v2': resnet_model.create_resnet101_v2_model,
   # 'resnet152': resnet_model.create_resnet152_model,
   # 'resnet152_v2': resnet_model.create_resnet152_v2_model,
}


def _get_model_map(dataset_name): 
    return _model_name_to_imagenet_model 
    
def get_model_config(model_name, dataset): 
  """Map model name to model network configuration.""" 
  model_map = _get_model_map(dataset.name) 
  return model_map[model_name]() 
