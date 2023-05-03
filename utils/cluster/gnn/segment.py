from torchvision.utils import draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from mat_extract import descriptor_mat
from torch_geometric.data import Data
from extractor import ViTExtractor
from gnn_pool import GNNpool
import torch.optim as optim
import numpy as np
import torch
import util
import cv2
import os


def GNN_seg(epoch, step, K, pretrained_weights, dir, cc, b_box, log_bin, res, facet, layer, stride, device):
    """
    Apply GNN segmentation for images in specified dir, will show figure on screen interactively,
    Every image will pass 10 epochs in the GNN and display results, this will happen recursively
    For n steps for every image.
    @param epoch: Number of epochs for every step in image
    @param step: Number of steps for Every image
    @param K: Number of segments to search in each image
    @param pretrained_weights: Weights of pretrained images
    @param dir: Input images directory
    @param cc: If k==2 chose the biggest component, and discard the rest (only available for k==2)
    @param b_box: If true will output bounding box (for k==2 only), else segmentation map
    @param log_bin: Apply log binning to the descriptors (correspond to smother image)
    @param res: Resolution for dino input
    @param facet: Facet to extract descriptors
    @param layer: Layer to extract descriptors
    @param stride: Stride for VIT descriptor extraction
    @param device: Device to use ('cuda'/'cpu')
    """
    # We init models once, and not every other file
    ##########################################################################################
    # Dino model init
    extractor = ViTExtractor('dino_vits8', stride, model_dir=pretrained_weights, device=device)
    # VIT small feature dimension
    if not log_bin:
        feats_dim = 384
    else:
        feats_dim = 6528

    # GNN model init
    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    ##########################################################################################
    plot_data = []
    for filename in os.listdir(dir):
        # If not image, skip
        if not filename.endswith((".jpg", ".png")):
            continue

        # loading images
        image_tensor, org_image = util.load_data_img(dir + filename, res)
        # Extract deep features, and create an adj matrix
        W, F, D = descriptor_mat(image_tensor, extractor, layer, facet, bin=log_bin ,device=device)

        adj = W
        feats = F
        image = org_image

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(adj, feats)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        for st in range(step):
            for _ in range(epoch):
                opt.zero_grad()
                A, S = model(data, torch.from_numpy(adj).to(device))
                loss = model.loss(A, S)
                loss.backward()
                opt.step()

            # polled matrix (after softmax, before argmax)
            S = S.detach().cpu()
            S = torch.argmax(S, dim=-1)

            # Reshape pooled graph
            # -1 is needed only for stride==4 of descriptor extraction
            S = np.array(torch.reshape(S, ((image_tensor.shape[2] // 4) - 1, int(image_tensor.shape[3] // 4) - 1)))

            # check if background is 0 and main object is 1 in segmentation map
            if (S[0][0] + S[S.shape[0] - 1][0] + S[0][S.shape[1] - 1] + S[S.shape[0] - 1][S.shape[1] - 1]) > 2:
                S = 1 - S

            # chose largest component (for k == 2)
            if cc:
                mask = util.largest_cc(S)
            else:
                mask = S

            # Resize mask to original image size
            mask = cv2.resize(mask.astype('float'), (image[:, :, 0].shape[1], image[:, :, 0].shape[0]),
                              interpolation=cv2.INTER_NEAREST)

            # Apply bounding box to largest component (for k == 2)
            if b_box:
                box = masks_to_boxes(torch.from_numpy(mask).unsqueeze(0))
                image = draw_bounding_boxes(torch.from_numpy(np.einsum('ijk->kij', image)), box, colors='red').numpy()
                image = np.einsum('kij->ijk', image)
            plot_data.append(image)
            plot_data.append(mask)

    # Plot segmentation and original image
    util.im_show_n(plot_data, 2, 'Results')


if __name__ == '__main__':
    ################################################################################
    # GNN parameters
    ################################################################################
    # Numbers of epochs per step
    epochs = 10
    # Number of steps per image
    step = 1
    # Number of clusters
    K = 2
    ################################################################################
    # Processing parameters
    ################################################################################
    # Show only largest component in segmentation map (for k == 2)
    cc = True
    # show bounding box (for k == 2)
    b_box = True
    # Apply log binning to extracted descriptors (correspond to smoother segmentation maps)
    log_bin = False
    ################################################################################
    # Descriptors extraction parameters
    ################################################################################
    # Directory to pretrained Dino
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    # Resolution for dino input, higher res != better performance as Dino was trained on (224,224) size images
    res = (224, 224)
    # stride for descriptor extraction
    stride = 4
    # facet for descriptor extraction (key/query/value)
    facet = 'key'
    # layer to extract descriptors from
    layer = 11
    ################################################################################
    # Data parameters
    ################################################################################
    # Directory of image to segment
    dir = './images/notebook/'
    ################################################################################
    # Chose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If Dino Directory doesn't exist than download
    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)

    GNN_seg(epochs, step, K, pretrained_weights, dir, cc, b_box, log_bin, res, facet, layer, stride, device)
