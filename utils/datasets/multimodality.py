"""
===============
Custom datasets
===============

Created 2022/05/04
@author: Chen Solomon
@author: Michal Katirai
@organization: Weizmann Institute of Science
"""
# imports
import abc
from pydicom import dcmread
import numpy as np
import os
from torchvision.datasets import VisionDataset
from typing import Callable, Optional


class MuMoDataset(VisionDataset, abc.ABC):
    """
    Abstract Base Class for Dataset classes used for Multi-Modality project
    """

    def __init__(
            self,
            root: str,
            targets_root: str = '',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        """

        @param root:  Directory to data
        @param targets_root: directory to targets / labels
        @param transform: transform to apply on samples
        @param target_transform: transform to apply on targets
        """
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )
        if len(targets_root) == 0:
            self.targets_root = root
        else:
            self.targets_root = targets_root

    @abc.abstractmethod
    def load(self) \
            -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Abstract method to load the data
        @return:
        """
        return [np.empty(0)], [np.empty(0)], [np.empty(0)], [np.empty(0)]

    def __len__(self) -> int:
        return self.n_samples


class DatasetMGMRUS(MuMoDataset):
    """
    Abstract base class for loading MG, MR and US modalities
    """

    # TODO: implement Multi-process data loading (https://pytorch.org/docs/stable/data.html#multi-process-data-loading)
    def __init__(
            self,
            root='',
            targets_root: str = '',
            target_transform: Optional[Callable] = None,
            mg_transform: Optional[Callable] = None,
            mr_transform: Optional[Callable] = None,
            us_transform: Optional[Callable] = None,
    ):
        """
        @param root: directory for loading data
        @param targets_root: directory for loading data labels (targets)
        @param target_transform: transform for targets when queried
        @param mg_transform: transform for MG images when queried
        @param mr_transform: transform for MR images when queried
        @param us_transform: transform for US images when queried
        """
        super().__init__(
            root=root,
            targets_root=targets_root
        )
        # save transforms as attributes
        self.target_transform = target_transform
        self.mg_transform = mg_transform
        self.mr_transform = mr_transform
        self.us_transform = us_transform
        # load data
        self.target, self.mg, self.mr, self.us = self.load()
        # get data length
        assert len(self.target) == len(self.mg) \
               and\
               len(self.mg) == len(self.mr)\
               and\
               len(self.mr) == len(self.us), 'numbers of scans are inconsistent'
        self.n_samples = len(self.target)

    def samples_transform(self, target, mg, mr, us):
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.mg_transform is not None:
            mg = self.mg_transform(mg)
        if self.mr_transform is not None:
            mr = self.mr_transform(mr)
        if self.us_transform is not None:
            us = self.us_transform(us)
        return target, mg, mr, us

    def __getitem__(self, item: int) -> tuple:
        # Get items
        target, mg, mr, us = self.target[item], self.mg[item], self.mr[item], self.us[item]
        # Apply transforms
        target, mg, mr, us = self.samples_transform(target, mg, mr, us)
        return target, mg, mr, us

    def __iter__(self):
        return iter(zip(self.target, self.mg, self.mr, self.us))


class MuMoPublicDataset(DatasetMGMRUS):
    """
    Dataset class used for Multi-Modality project question 1.
    Question 1: Classification: Estimate chances of malignancy / BI-RADS
    """

    def load(self):
        """
        @return: tuple targets, mg, mr, us. Each one of them is a list, where each element (numpy array)
        represents data of different patient
        """
        mr, us, mg = [], [], []
        targets = np.array([[0], [0], [1]])
        for folder in os.listdir(self.root):  # loop over patients
            patient_path = self.root + "/" + folder
            im_list = os.listdir(patient_path)
            # TODO: Add explanation to key definition
            get_img_idx = lambda f: int(''.join(filter(str.isdigit, f)))
            im_list.sort(key=get_img_idx)
            slice_location = []
            mr_per_patient, us_per_patient, mg_per_patient = [], [], []
            for image in im_list:
                im_path = patient_path + "/" + image
                ds = (dcmread(im_path))
                ds_array = ds.pixel_array.astype(dtype=np.float32)
                modality = ds['Modality'].value  # finding modality type
                if modality == "MG":
                    mg_per_patient = ds_array  # TODO: handle multiple MG images
                elif modality == "MR":
                    mr_per_patient.append(ds_array)
                    slice_location.append(float(ds['SliceLocation'].value))
                elif modality == "US":
                    us_per_patient = ds_array  # TODO: handle multiple US images
            sorted_mr = []
            for _, x in sorted(zip(slice_location, mr_per_patient)):
                sorted_mr.append(x)
            mg.append(mg_per_patient)
            mr3d = np.stack(sorted_mr)  # converting MRI's slices to 3D tensor
            mr.append(mr3d[:, :, :])  # making the same amount of slices to each patient
            us.append(us_per_patient.mean(axis=-1))
        return targets, mg, mr, us