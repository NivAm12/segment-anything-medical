"""
=============================
Ultrasonography Augmentations
=============================

Created 2022/06/27
@author: Chen Solomon
@organization: Weizmann Institute of Science
"""
from abc import abstractmethod
from albumentations.augmentations.geometric import transforms as aag_transforms
import torchvision.transforms as transforms
from scipy import ndimage
import numpy as np
import random
import torch
from ..builder.registry import TORCHVISION_TRANS, ALBU_TRANS


"""
Reasonable basic augmentations to use:
    2022/06/28
    translation / rotation / scaling / affine (edges need to be cropped afterwards)
    crop (on rectangular US)
    horizontal flip (on NON-abdominal US, e.g., breast, cranial, ovary)
    Gaussian blur / Sharpen ?
    Additive white Gaussian noise: for channel / IQ data?
    Multiplicative Rayleigh noise: common model for Bmode speckle noise (see DOI: 10.1109/JSTSP.2020.3001829)
    Deformation field (maybe using simpleitk?)
    2022/06/30
    Shadow (should be specific to US)
    
    Additionally: Ultrasound images augmentation repo from MICCAI 2021 (Note that the main branch is empty)
        https://github.com/mariatirindelli/UltrasoundAugmentation/tree/release/miccai2021
        https://miccai2021.org/openaccess/paperlinks/2021/09/01/402-Paper2428.html
    
"""


# Register transforms
ALBU_TRANS.register_class(aag_transforms.ElasticTransform)
TORCHVISION_TRANS.register_class(transforms.RandomErasing)


@TORCHVISION_TRANS.register_class
class USNoiseSpeckle:
    """
    Multiplicative Speckle (modified Rayleigh) + Additive Noise (Gaussian)
    Model is taken from "Rayleigh-maximum-likelihood bilateral filter for ultrasound image enhancement"
    https://doi.org/10.1186/s12938-017-0336-9
    """

    def __init__(self, speckle_scale=0.2, noise_loc=0, noise_scale=0.1, noise_depth_depend=0):
        self.speckle_scale = speckle_scale
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale
        self.noise_depth_depend = noise_depth_depend

    def __call__(self, img):
        # Apply median filter on image
        img_med = ndimage.median_filter(img, 3)
        # Compute speckle
        speckle = np.random.rayleigh(scale=self.speckle_scale, size=img.shape)
        speckle = np.max(1 - self.speckle_scale + speckle, 0)[np.newaxis, :]
        speckle = speckle.astype(img_med.dtype)
        # Compute noise
        noise = np.random.normal(loc=self.noise_loc, scale=self.noise_scale, size=img.shape)
        noise = noise.astype(img_med.dtype)
        noise = img.mean() * noise
        # Compute noise depth dependence
        # depth_axis = np.arange(img.shape[-2])[:, np.newaxis]
        # noise_depth_modifier = np.exp(self.noise_depth_depend * depth_axis)
        noise_depth_modifier = np.logspace(0, self.noise_depth_depend, img.shape[-2])[:, np.newaxis]
        noise_depth_modifier = noise_depth_modifier.astype(img_med.dtype)
        noise = noise * noise_depth_modifier
        # Apply noise and speckle
        out = noise + img_med * speckle
        return out


@TORCHVISION_TRANS.register_class
class RandomCroppedRotation:
    """
    Apply rotation and crop blank margins
    Compatible with torch.Tensor types only
    """

    def __init__(self, **kwargs):
        """
        Initialize rotation transform
        :param kwargs: arguments for definition of rotation transform
        """
        self.rotation_transform = transforms.RandomRotation(**kwargs)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Rotate
        rotated_image = self.rotation_transform(img)
        # Crop
        cropped_image = self.remove_edges(rotated_image=rotated_image, original_image=img)
        return cropped_image

    def remove_edges(self, rotated_image: torch.Tensor, original_image: torch.Tensor = None) -> torch.Tensor:
        """
        removes black edges of transformed image
        Assumes center rotation
        @param rotated_image: image to be cropped
        @param original_image: original image
        @return:
        """
        image_template = self.get_2d_image_template(rotated_image)
        if original_image is not None:
            orig_image_temp = self.get_2d_image_template(original_image)
            if \
                    torch.nonzero(orig_image_temp[0, :]).nelement() == 0 \
                        or \
                        torch.nonzero(orig_image_temp[-1, :]).nelement() == 0 or \
                        torch.nonzero(orig_image_temp[:, 0]).nelement() == 0 or \
                        torch.nonzero(orig_image_temp[:, -1]).nelement() == 0:
                return rotated_image
        # get indices across x axis
        x_axis = torch.arange(image_template.size(-2))
        # get y indices of diagonals of the image
        aspect_ratio = image_template.size(-2) / image_template.size(-1)
        y_axis_diag_0 = (x_axis / aspect_ratio).round().long()
        # y_axis_diag_1 = image.size(-1) - 1 - y_axis_diag_0
        # Get diagonal values
        diag_0 = image_template.squeeze()[x_axis[:], y_axis_diag_0[:]]
        # diag_1 = image.squeeze()[x_axis[:], y_axis_diag_1[:]]
        # Get nonzero element indices for diagonal
        x_axis_nnz_0 = diag_0.nonzero()
        # x_axis_nnz_1 = diag_1.nonzero()
        # Use diagonal to determine crop
        x_crop = torch.tensor([x_axis_nnz_0.min(), x_axis_nnz_0.max()])
        y_crop = y_axis_diag_0[x_crop]
        cropped_image = transforms.functional.crop(
            rotated_image,
            top=x_crop[0],
            left=y_crop[0],
            height=x_crop[1] - x_crop[0],
            width=y_crop[1] - y_crop[0]
        )
        return cropped_image

    @staticmethod
    def get_2d_image_template(img):
        """
        Get 2D image to characterise image with less dimensions
        :param img:
        :return:
        """
        if 4 >= img.ndim and img.ndim >= 2:
            image_template = img
            # project dimensions
            for idx_dim in range(2, img.ndim):
                image_template = torch.max(image_template.abs(), dim=0)[0]
        else:
            raise NotImplementedError('only suppots 2- to 4-D arrays')
        return image_template


class NakagamiAcousticShadow:
    """
    used for adding acoustic shadow with Nakagami distribution in a Ultrasonography scan of unspecified form
    based on implementation at: https://github.com/rsingla92/speckle_n_shadow/blob/master/shadow.ipynb
    Created 08082022
    @author: Chen Solomon
    """
    
    def __init__(self, p: float = 1., nakagami_shape: float = 1, nakagami_spread: float = 1) -> None:
        self.p = p
        self.nakagami_shape = nakagami_shape
        self.nakagami_spread = nakagami_spread

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            mask = self._create_mask(img=img)
            noise = self._create_noise(img=img)
            output = mask * noise + (1 - mask) * img
        else:
            output = mask
        return output

    def _create_noise(self, img: torch.Tensor) -> torch.Tensor:
        """
        Nakagami distribution is generated from gamma distribution according to Wikipedia tip at:
        https://en.wikipedia.org/wiki/Nakagami_distribution
        TODO: find better documented method for generation of Nakagami distribution
        """
        gamma_concentration = torch.ones(img.shape) * self.nakagami_shape
        gamma_rate = torch.ones(img.shape) * self.nakagami_shape / self.nakagami_spread
        gamma_dist = torch.distributions.gamma.Gamma(concentration=gamma_concentration, rate=gamma_rate)
        gamma_samp = gamma_dist.sample()
        nakagami_noise = torch.sqrt(gamma_samp)
        return nakagami_noise

    @abstractmethod
    def _create_mask(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Invoked abstract method of class ' + self.__class__.__name__)


@TORCHVISION_TRANS.register_class
class RectNakagamiAcousticShadow(NakagamiAcousticShadow):
    def __init__(
            self,
            p: float = 1,
            nakagami_shape: float = 1,
            nakagami_spread: float = 1,
            n_shadows: int = 1,
            shadow_width: int = 45,
            shadow_height: int = None,
            darkness: float = 0.5,
    ) -> None:
        """
        used for adding acoustic shadow with Nakagami distribution in a Rectangular Ultrasonography scan
        based on implementation at: https://github.com/rsingla92/speckle_n_shadow/blob/master/shadow.ipynb
        Created 08082022
        @author: Chen Solomon
        """
        super(self.__class__, self).__init__(p=p, nakagami_shape=nakagami_shape, nakagami_spread=nakagami_spread)
        self.n_shadows = n_shadows
        self.shadow_width = shadow_width
        if shadow_height is None:
            self.shadow_height = shadow_width
        else:
            self.shadow_height = 2 * shadow_height
        self.darkness = darkness

    def _create_mask(self, img: torch.Tensor) -> torch.Tensor:
        # randomly choose points of origin for shadow
        # Use binomial distribution randomize number of origins
        # probability of origin is proportional to grayscale level
        origin_prob = img / img.sum()
        # use binomal distribution to control average number of shadows
        origin_dist = torch.distributions.binomial.Binomial(self.n_shadows, origin_prob)
        origin_map = torch.minimum(origin_dist.sample(), torch.tensor([1]))
        # widen shadow origin
        origin_kernel_x = torch.ones((1, 1, 1, self.shadow_width))
        origin_map_wide = torch.nn.functional.conv2d(origin_map, origin_kernel_x, padding=(0, self.shadow_width // 2))
        # keep dimensions in case of even shadow width
        if self.shadow_width % 2 == 0:
            origin_map_wide = origin_map_wide.index_select(dim=-1, index=torch.arange(0, origin_map_wide.shape[-1] - 1))
        # continue shadow downwards without the origin
        origin_kernel_z = torch.ones((1, 1, self.shadow_height, 1))
        shadow_temp = torch.nn.functional.conv2d(
            origin_map_wide,
            origin_kernel_z,
            padding=(
                self.shadow_height // 2,
                0
            )
        )
        # keep dimensions in case of even shadow width
        if self.shadow_height % 2 == 0:
            shadow_temp = shadow_temp.index_select(dim=-1, index=torch.arange(0, shadow_temp.shape[-1] - 1))
        # make sure values do not exceed 1
        shadow_temp = torch.minimum(shadow_temp, torch.tensor([1]))
        # shadow_temp = torch.minimum(torch.cumsum(origin_map_wide, dim=-2) - origin_map_wide, torch.tensor([1]))  # OLD
        # Make sure that mask does not shadow above the origin
        shadow_one_sided = torch.minimum(torch.cumsum(origin_map_wide, dim=-2), torch.tensor([1]))
        one_sided_width = 4 * self.shadow_width - 1  # keeps this odd to save cropping for even kernels
        one_sided_kernel_x = torch.ones((1, 1, 1, one_sided_width))
        shadow_one_sided_wide = torch.nn.functional.conv2d(shadow_one_sided, one_sided_kernel_x, padding=(0, one_sided_width // 2))
        shadow_one_sided_wide = torch.minimum(shadow_one_sided_wide, torch.tensor([1]))
        # Randomize "shadowed" pixels according to distribution
        shadow_prob = shadow_temp * shadow_one_sided_wide * self.darkness
        shadow_dist = torch.distributions.bernoulli.Bernoulli(shadow_prob)
        shadow_bin = shadow_dist.sample()
        # Blur mask
        shadow_blurred = transforms.GaussianBlur(
            kernel_size=(
                self.shadow_width, self.shadow_width
            ),
            sigma=(
                self.shadow_width // 2, self.shadow_width // 2
            ))(shadow_bin)
        return shadow_blurred