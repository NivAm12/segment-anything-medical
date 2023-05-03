"""
=====================================
Transforms Built from Config files
=====================================

Created 2022/07/17
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""
import albumentations as albu
import numpy as np
import torch
from torchvision import transforms
# local imports
from .registry import ALBU_TRANS
from .builder import ObjectFromConfig
from .registry import TORCHIO_TRANS
from .registry import TORCHVISION_TRANS
from .registry import TRANSFORMS
from .registry import US_Transforms
from typing import Union
from ..preprocess.mammography import mg_remove_edges

"""
Registry instances
"""

# register torchvision transforms
TORCHVISION_TRANS.register_class(transforms.Normalize)
TORCHVISION_TRANS.register_class(transforms.Resize)
TORCHVISION_TRANS.register_class(transforms.ToTensor)
TORCHVISION_TRANS.register_class(transforms.Grayscale)


class TransformFromConfig(ObjectFromConfig):
    registry = TRANSFORMS

    @staticmethod
    def _build_sequence(objects_list):
        raise NotImplementedError('Generic Transforms cannot be automatically composed')


@TRANSFORMS.register_class
@TORCHVISION_TRANS.register_class
class ToviTransformFromConfig(ObjectFromConfig):
    """
    Torchvision Transforms
    """
    registry = TORCHVISION_TRANS

    @staticmethod
    def _build_sequence(transforms_list):
        """
        Compose a sequence of transforms
        :return:
        """
        composed_transform = transforms.Compose(transforms_list)
        return composed_transform

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('factory class instance should not be called directly')


@TRANSFORMS.register_class
class AlbuTransformFromConfig(ObjectFromConfig):
    """
    Albumentation Transforms
    """
    registry = ALBU_TRANS

    @staticmethod
    def _build_sequence(transforms_list):
        """
        Compose a sequence of transforms
        :return:
        """
        composed_transform = albu.Compose(transforms_list)
        return composed_transform

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('factory class instance should not be called directly')


@TRANSFORMS.register_class
@TORCHVISION_TRANS.register_class
class ToviWrapper4Albu:
    """
    albu transform with modified interface that is compatible with torchvision
    """
    def __init__(self, *args, **kwargs):
        """
        initialize albumentations transform
        :param args:
        :param kwargs:
        """
        self.transform = AlbuTransformFromConfig(*args, **kwargs)

    def __call__(self, image, mask=None):
        if mask is None:
            out_image = torch.tensor(self.transform(image=np.array(image.squeeze()))['image'])
            out_image = torch.reshape(out_image, image.shape)
            return out_image
        else:  # resort to default api
            out_dict = self.transform(image=image, mask=mask)
            return out_dict


@TRANSFORMS.register_class
@TORCHVISION_TRANS.register_class
class ToviWrapper4AlbuMG:
    """
    albumentation transform with modified interface that is compatible with torchvision for mammography images
    """
    def __init__(self, *args, **kwargs):
        """
        initialize albumentations transform
        :param args:
        :param kwargs:
        """
        self.transform = AlbuTransformFromConfig(*args, **kwargs)

    def __call__(self, image, mask=None):
        out_image = torch.tensor(self.transform(image=np.array(image.squeeze()))['image']).unsqueeze(dim=0)
        out_image = mg_remove_edges(out_image)
        return out_image


@TRANSFORMS.register_class
@US_Transforms.register_class
class ClassWrapper4Func:
    """
    Used to wrap useful functional transforms to be used as part of Compose
    Assumes function receives and returns numpy.ndarray
    for specific API one needs to inherit from this class and wrap the API in the __call__ method
    (
    API examples
        torchvision.transforms:
            input: torch.Tensor
            output: torch.Tensor
        albumentations:
            input: numpy.ndarray
            output: {
                'image': numpy.ndarray,
                'mask': numpy.ndarray
                }
    )
    """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.function(*args, *self.args, **kwargs, **self.kwargs)
