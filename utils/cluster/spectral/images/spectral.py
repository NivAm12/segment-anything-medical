from scipy.sparse.linalg import eigsh
from extractor import ViTExtractor
from sklearn.cluster import KMeans
import torch.nn.functional as TF
from PIL import Image
from tqdm import tqdm
import urllib.request
import numpy as np
import mat_extract
import torch
import scipy
import util
import time
import os

def spectral_decomp(W, D, K, img, adaptive):
    
    D = scipy.sparse.diags(D)
    # Compute eigenvectors
    try:
        eigenvalues, eigenvectors = eigsh(D - W, k=(K + 1), sigma=0, which='LM', M=D)
    except:
        eigenvalues, eigenvectors = eigsh(D - W, k=(K + 1), which='SM', M=D)
    # swap axes to (vec_num, vec feat)
    eigenvalues = torch.from_numpy(eigenvalues)
    eigenvectors = torch.from_numpy(eigenvectors.T).float()


    # Resolve sign ambiguity
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]


    P = 8 # patch size
    H_patch, W_patch = 55, 55
    H_pad, W_pad = H_patch * P, W_patch * P

    image_list = []
    for i in range(1, K + 1):
        eigenvector = eigenvectors[i].reshape(1, 1, H_patch, W_patch)  # .reshape(1, 1, H_pad, W_pad)
        eigenvector = TF.interpolate(eigenvector, size=(H_pad, W_pad), mode='bilinear', align_corners=False)  # slightly off, but for visualizations this is okay
        image_list.append(eigenvector.squeeze().numpy())

    seg = segmentations(eigenvalues, eigenvectors, K, H_patch, W_patch, adaptive = adaptive,non_adaptive_num_segments=K + 1)

    
    image_list.append(seg)
    image_list.append(img)

    util.im_show3(image_list, "egan decomp")


def segmentations(eigan_vals, eigan_vec, num_eigenvectors, H_patch, W_patch, adaptive= False, non_adaptive_num_segments= 6):
    # If adaptive, we use the gaps between eigenvalues to determine the number of 
    # segments per image. If not, we use non_adaptive_num_segments to get a fixed
    # number of segments per image.
    if adaptive:
        indices_by_gap = np.argsort(np.diff(eigan_vals.numpy()))[::-1]
        index_largest_gap = indices_by_gap[indices_by_gap != 0][0]  # remove zero and take the biggest
        n_clusters = index_largest_gap + 1
        # print(f'Number of clusters: {n_clusters}')
    else:
        n_clusters = non_adaptive_num_segments

    kmeans = KMeans(n_clusters=n_clusters)
    eigenvectors = eigan_vec[1:1+num_eigenvectors].numpy()  # take non-constant eigenvectors
    clusters = kmeans.fit_predict(eigenvectors.T)

    # Reshape
    if clusters.size == H_patch * W_patch:  # TODO: better solution might be to pass in patch index
        segmap = clusters.reshape(H_patch, W_patch)
    elif clusters.size == H_patch * W_patch * 4:
        segmap = clusters.reshape(H_patch * 2, W_patch * 2)
    else:
        raise ValueError()

    return segmap


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == '__main__':
    image_dir = './images/3.png'
    adaptive = False
    K = 5

    # Directory to pretrained Dino
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If Directory doesnt exist than download 
    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        download_url(url, pretrained_weights)
    
    # prepare image for pass in model
    image_tensor, img_vec, org_image = mat_extract.load_data((224,224), image_dir , device)
    # init model
    extractor = ViTExtractor('dino_vits8', stride=4, model_dir=pretrained_weights,device=device)
    # Extract descriptors from model
    W, F, D = mat_extract.descriptor_seg(image_tensor, img_vec, org_image, extractor, 4, layer=11, facet='key', model_dir=pretrained_weights)

    # Apply spectral decomposition
    spectral_decomp(W, D, K, org_image, adaptive)