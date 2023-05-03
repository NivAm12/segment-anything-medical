"""
==================================
Optimizers Built from Config files
==================================

Created 2022/08/09
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""
import torch.optim
# local imports
from .registry import LRSCHEDULER
from .builder import ObjectFromConfig
from .registry import OPTIMIZER

# Optimizers: register relevant classes
OPTIMIZER.register_class(torch.optim.SGD)
OPTIMIZER.register_class(torch.optim.Adam)

# Learning rate scheduler: register relevant classes
LRSCHEDULER.register_class(torch.optim.lr_scheduler.LambdaLR)
LRSCHEDULER.register_class(torch.optim.lr_scheduler.ExponentialLR)


class OptimizerFromConfig(ObjectFromConfig, torch.optim.Optimizer):
    """
    Optimizer type to be built from config file and used for training
    Note:
        currently only method accessible from instances of class (without accessing the optimizer attribute) are step and zero_grad
    """
    registry = OPTIMIZER

    @staticmethod
    def _build_sequence(_):
        raise TypeError('Optimizer builder does not support list of configs')


class LRSchedulerFromConfig(ObjectFromConfig):
    registry = LRSCHEDULER
    """
    Learning rate scheduler to be built from config file and used for training
    Note:
        currently only method accessible from instances of class without accessing the optimizer attribute is step() 
    """

    @staticmethod
    def _build_sequence(_):
        raise TypeError('Learning rate scheduler builder does not support list of configs')
