# Builder - Create class instances from YAML configuration files

The code in this folder is meant  to provide a flexible way of running machine learning (DL) training in PyTorch, using configuration files. The usage of configuration files enables running the same training session on a different model, using different augmentations / optimizer / preprocessing without editing the training code itself (i.e., this prevents your training setup from being [Hard Coded](https://stackoverflow.com/questions/1895789/what-does-hard-coded-mean), and separates the implementation from the parameters it takes as an input). This can be useful when running multiple experiments on the same code using different models and trying to find which work better.

## Example for using the builder
```Python
from torch import nn
from samplutils.builder import model as model_builder
from samplutils.builder.registry import Registry


# define a new registry: an object that is used to map class name to the class itself 
NETWORK_REGISTRY = Registry("Neural Networks")


# register a model class in your registry
@NETWORK_REGISTRY.register_class
class FullyConnectedNet(nn.Linear):
    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)


# define a config for the model you want to create
network_class2_config = {
    'type': 'FullyConnectedNet',
    'in_features': 1,
    'out_features': 1
}
# use the builder, config and registry to get the wanted model 
fully_connected_net = model_builder.ModelFromConfig(
    network_class2_config, 
    registry=NETWORK_REGISTRY
)
print('The registry used is: \n')
print(NETWORK_REGISTRY)
print('\nThe model created is: \n')
print(fully_connected_net)
```
- New objects can be registered and used with builder by applying the ```register_class``` method of the relevant registry on them (either directly or using a decorator like shown in the example).
- The ```registry``` module contains some useful registries that are defined as the default registries for the different builders.
- The different builders are defined within the ```model```, ```optimizer``` and ```transforms``` modules.
- Note that the builders in the ```transforms``` module are more diverse and contains wrappers for different types of transforms.
