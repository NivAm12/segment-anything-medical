"""
==============================
Models Built from Config files
==============================

Created 2022/06/21
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""
from torch import nn
from typing import Sequence

# local imports
from .registry import LOSSES
from .registry import MODELS
from .builder import ObjectFromConfig

# Basic Example for flexible model configuration
# >> registry = {'unet': UnetModelClass}  # registry is a dictionary-like object mapping names to classes
# >> net_name = 'unet'  # class names we get from a config file .json/.yml
# >> net_class = registry[net_name]  # -> maps to model class
# >> model_instance = net_class(**arguments_in_dict_defined_elsewhere)

"""
Registry instances
"""
LOSSES.register_class(nn.CrossEntropyLoss)


class ModelFromConfig(ObjectFromConfig, nn.Module):
    registry = MODELS
    """
    Model that is initialized using a config file
    """

    @staticmethod
    def _build_sequence(module_list: Sequence[nn.Module]):
        """

        :param module_list: list containing models to be added
        :return:
        """
        sequence_model = nn.Sequential(*module_list)
        return sequence_model


class LossFromConfig(ModelFromConfig):
    registry = LOSSES
    """
    Loss that is built from config
    """
