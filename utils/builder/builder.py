"""
=========================================
Build Class instances from Config objects
=========================================

Created 2022/06/21
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""
import inspect
import os
from .registry import Registry
from typing import Union
import yaml


# Basic Example for flexible model configuration
# >> registry = {'unet': UnetModelClass}  # registry is a dictionary-like object mapping names to classes
# >> net_name = 'unet'  # class names we get from a config file .json/.yml
# >> net_class = registry[net_name]  # -> maps to model class
# >> model_instance = net_class(**arguments_in_dict_defined_elsewhere)


class ObjectFromConfig:
    """
    Abstract base class: Class instance that is initialized using a config
    Subclasses should implement the relevant methods for the objects they instantiate
    """
    # Class attribute that maps class name to class object
    registry = Registry('Objects')

    def __new__(
            cls,
            config: Union[str, dict, list],
            registry: Registry = None,
            **config_kwargs,
    ):
        """
        build new class from config and return
        @param config: dictionary or .yml file containing config (mapping)
        @param registry: mapping from augmentations class names to classes
        @config_kwargs: taking config parameters as key word arguments
        """
        # Set registry
        if registry is None:
            _registry = cls.registry
        else:
            _registry = registry
        # Set config (Handle config cases)
        if isinstance(config, str):  # load if path to config file
            assert os.path.isfile(config)
            with open(config, 'r') as f:
                _config = yaml.load(f)
        elif config is None:
            _config = {}
        else:
            _config = config
        # add to config if there are additional arguments
        if isinstance(_config, list):
            for cfg_ in _config:
                cfg_.update(config_kwargs)
        else:
            _config.update(config_kwargs)
        if isinstance(_config, list):
            objects_list = [cls._build_from_config(cfg_, _registry) for cfg_ in _config]
            new_class_instance = cls._build_sequence(objects_list)
        else:
            new_class_instance = cls._build_from_config(_config, _registry)
        return new_class_instance

    @staticmethod
    def _build_sequence(objects_list):
        """
        build a sequence of transforms
        @:param objects_list: list of objects to be added sequentially
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _build_from_config(config: dict, registry: Registry):
        """
        Build a class instance from config dict and registry object.
        config should contain a key named "type" that is the class name
        original @author: Tianwei Yin (https://github.com/tianweiy/CenterPoint/blob/master/det3d/utils/registry.py)
        Adapted by: Chen Solomon
        """
        args = config.copy()
        # check config type and that it contains class name
        assert isinstance(args, dict) and "type" in args
        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    "{} is not in the {} registry".format(obj_type, registry.name)
                )
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                "type must be a str or valid type, but got {}".format(type(obj_type))
            )
        return obj_cls(**args)
