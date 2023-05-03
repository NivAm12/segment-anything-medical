from extractor import ViTExtractor
import torch.nn.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse
import numpy as np
import utils
import torch
import util
import cv2
import os

def descriptor_seg(image_tensor, img_arr, org_image,extractor, stride, layer, facet, bin: bool = False,
          include_cls: bool = False, device='cuda', model_dir=None, color_data = True, color_mat = 'knn', color_alpha= 0.2):
    """

    @param image_tensor: Tensor of size (batch, height, width)
    @param seg_arr: numpy array of corrosponding seg to image tensor of size (batch * height * width, 1)
    @param model_type: model type, for example 'dino_vits8'
    @param stride: stride of the vit transformer (curretly == patchSize for t-SNE compitability)
    @param layer: Layer to extract the descriptors from
    @param facet: Facet to extract the descriptors from
    @param bin: apply log binning to the descriptor. default is False.
    @param include_cls: To include CLS tocken in extracted descriptor
    @param device: Training device
    @param epoch: Nunber of epoch calling to the function; used for animation
    @param model_dir: model dir to load is not using while training
    @return:
    """


    # images to descriptors.
    # input is a tensor of size batch X height X width,
    # output is size: batch X 1 X height/patchsize * width/patchsize X descriptor feature size
    descriptor = extractor.extract_descriptors(image_tensor.to(device), layer, facet, bin,include_cls).cpu().numpy()

    # batch X height/patchsize * width/patchsize X descriptor feature size
    descriptor = np.squeeze(descriptor, axis=1)

    # batch * height/patchsize * width/patchsize X descriptor feature size
    descriptor = descriptor.reshape((descriptor.shape[0] * descriptor.shape[1], descriptor.shape[2]))

    # descriptor feature size X batch * (height* width/(patchsize ^2))
    F = descriptor

    # descriptors affinities matrix
    W = F @ F.T
    # threshold == 0
    W = W * (W > 0)

    #norm
    W = W / W.max()

    # combaining color data affinities 
    if(color_data):
            # Color affinities (of type scipy.sparse.csr_matrix)
            if color_mat == 'knn':
                W_lr = util.knn_affinity(img_arr / 255)
            elif color_mat == 'rw':
                W_lr = util.rw_affinity(img_arr / 255)
            
            # Convert to dense numpy array
            W_color = np.array(W_lr.todense().astype(np.float32))
            W = W + W_color * color_alpha

    # D is row wise sum diagonal of W
    D = W.dot(np.ones(W.shape[1], W.dtype))
    D[D < 1e-12] = 1.0  # Prevent division by zero. defualt threshold==1e-12
    
    # D - W == laplacian
    return W, F, D


def load_data(image_size, chosen_dir, device = 'cuda'):
    """
    @param device: Training device
    @param image_size: Resize image and seg to size
    @return:
    """

    # tensor vector of original images, and corresponding 

    image_tensor, image = utils.preprocess(os.path.join(chosen_dir), image_size)
    # lists to stacked arrays (batch X height X width)
    image_tensor = image_tensor
    image = np.array(image)

    # saving image without resizing
    org_image = image
    # divide to patchsize X patchsize patches and pool using knn (batch X height/patchsize X width/patchsize)
    image = cv2.resize(image, (55,55), interpolation = cv2.INTER_AREA)

    return image_tensor, image, org_image


if __name__ == '__main__':
    # Params
    #######################################################################
    #######################################################################

    # Resize image.
    image_size = (224, 224)

    # pretrained model, ckpt dir
    pretrained_weights = "./checkpoint.pth"

    # Vit architecture.  choices=['vit_tiny', 'vit_small', 'vit_base']
    arch = 'dino_vits8'

    stride = 4

    layer = 11

    facet = 'key'

    #######################################################################
    #######################################################################

    # gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for filename in tqdm(os.listdir('./images')):
        if not filename.endswith((".png", ".jpg")):
            continue
        # loading images
        image_tensor, img_vec, org_image = load_data(image_size, "./images/" + filename , device)

        extractor = ViTExtractor(arch, stride, model_dir=pretrained_weights,device=device)

        W, F, D = descriptor_seg(image_tensor, img_vec, org_image, extractor, 4, layer, facet, model_dir=pretrained_weights)

        np.save('./mat/adj/' + filename + '_vits8_stride=4.npy', W)
        np.save('./mat/feats/' + filename + '_vits8_stride=4.npy', F)
        np.save('./mat/diag/' + filename + '_vits8_stride=4.npy', D)