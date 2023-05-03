import torch
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def mean_and_std(image_dir):
# calculate mean and std of complete dataset.
# If all images in the same size you can from batch size from 1 for speedup 
    dataset = datasets.ImageFolder(image_dir, ToTensor())

    image_loader = DataLoader(dataset,
                              batch_size=1,
                              shuffle=False)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(image_loader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    # output
    print('mean: ' + str(mean))
    print('std:  ' + str(std))

if __name__ == '__main__':

    dir = 'Y:/SAMPL_training/public_datasets/datasets/US/Thyroid/stanford_aimi/images'
    mean_and_std(dir)