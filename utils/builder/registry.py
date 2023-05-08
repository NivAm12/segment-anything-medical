"""
===================================================
Register classes and Map class name to class object
===================================================

Created 2022/06/21
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""
import inspect
from typing import Any, Type


# Basic Example for flexible model configuration
# >> registry = {'unet': UnetModelClass}  # registry is a dictionary-like object mapping names to classes
# >> net_name = 'unet'  # class names we get from a config file .json/.yml
# >> net_class = registry[net_name]  # -> maps to model class
# >> model_instance = net_class(**arguments_in_dict_defined_elsewhere)

class Registry(object):
    """
    A class to register all object types in
    original @author: Tianwei Yin (https://github.com/tianweiy/CenterPoint/blob/master/det3d/utils/registry.py)
    Adapted by: Chen Solomon

    How to use:
    Each class used in the project should be decorated with the proper registry's 'register_module' method
    The corresponding registry object (defined in the corresponding builder's file) should be imported in the .py file
    The object's .py file should be imported in the __init__.py file of its python package (folder)

    Example:
    # >>> from registry import Registry
    # >>>
    # >>> BLOCKS = Registry("Blocks")
    # >>>
    # >>> @BLOCKS.register_class
    # ... class NeuralNet: pass
    # >>>
    # >>> print(BLOCKS)
    Registry(name=Blocks, items=['NeuralNet'])
    """

    def __init__(self, name):
        """

        @param name: identifier of registry type
        """
        self._name = name
        self._class_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(
            self._name, list(self._class_dict.keys())
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def class_dict(self):
        return self._class_dict

    def get(self, key):
        return self._class_dict.get(key, None)

    def _register_class(self, cls: Type[Any]):
        """Register a module.
        @param: module_class (:obj:`nn.Module`): Module to be registered.
        """
        # check that input is a class
        if not inspect.isclass(cls):
            raise TypeError(
                "input must be a class, but got {}".format(type(cls))
            )
        # check that class name is not already registered
        module_name = cls.__name__
        if module_name in self._class_dict:
            raise KeyError(
                "{} is already registered in {}".format(module_name, self.name)
            )
        self._class_dict[module_name] = cls

    def register_class(self, cls: Type[Any]) -> Type[Any]:
        self._register_class(cls)
        return cls

    def __len__(self):
        return self._class_dict.__len__()


"""
Registry Instances
"""
# # torch.nn.Module
# Registry of generic models
MODELS = Registry("MuMO_Models")

# Loss registry
LOSSES = Registry('Loss Functions')

# Optimizers registry
OPTIMIZER = Registry('Optimizers')

# Learning rate scheduler:
LRSCHEDULER = Registry('Learning Rate Schedulers')

# Registry of augmentations
TRANSFORMS = Registry("Transforms")
TORCHVISION_TRANS = Registry("Transforms from torchvision.transforms")
US_Transforms = Registry("Ultrasound Transforms")
ALBU_TRANS = Registry("Transforms from albumentations")
TORCHIO_TRANS = Registry("Transforms from torchio")

# Registry of datasets
DATASETS = Registry("Datasets")
