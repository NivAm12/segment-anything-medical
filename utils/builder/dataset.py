"""
===============================
Dataset Built from Config files
===============================

Created 2022/11/10
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""
# local imports
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from .builder import ObjectFromConfig
from .registry import DATASETS


class DatasetFromConfig(ObjectFromConfig, Dataset):
    registry = DATASETS
    """
    Dataset that is built from config
    """

    def __getitem__(self, index) -> T_co:
        return Dataset.__getitem__(self, index)

    @staticmethod
    def _build_sequence(_):
        raise TypeError('Dataset builder does not support list of configs')
