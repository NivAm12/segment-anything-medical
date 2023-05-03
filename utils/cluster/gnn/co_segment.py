from torchvision.transforms.functional import resized_crop
from torchvision.ops import masks_to_boxes
from mat_extract import descriptor_mat
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from torchvision import transforms
from extractor import ViTExtractor
from gnn_pool import GNNpool
import torch.optim as optim
import numpy as np
import torch
import util
import cv2
import os


def GNN_seg(epoch, K, pretrained_weights, dir, cc, log_bin, res, facet, layer, stride, device):
    """
    Apply co-segmentation for input images. Only available for k == 2 for now.
    @param epoch: Number of epochs for every step in image
    @param K: Number of segments to search in each image
    @param pretrained_weights: Weights of pretrained images
    @param dir: Directory for input images
    @param cc: chose the biggest component, and discard the rest of the segmentation map
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
    # counter for processed images
    count = 1
    # original images
    org = []
    # segmented images
    seg = []
    # cropped images
    embeds = []

    filenames = os.listdir(dir)
    for filename in filenames:
        # If not image, skip
        if not filename.endswith((".jpg", ".png")):
            continue

        # loading images
        image_tensor, org_image = util.load_data_img(dir + filename, res)
        # Extract deep features, and create an adj matrix
        W, F, _ = descriptor_mat(image_tensor, extractor, layer, facet, bin=log_bin, device=device)

        adj = W
        feats = F
        image = org_image

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(adj, feats)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model.load_state_dict(torch.load('./dict.pt', map_location=torch.device(device)))
        opt = optim.Adam(model.parameters(), lr=0.001)

        # run for specified epochs
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
        box = masks_to_boxes(torch.from_numpy(mask).unsqueeze(0))

        # Extract segmented areas and resize to (224,224) for input to dino trained transformer
        out = torch.swapaxes(
            resized_crop(torch.swapaxes(torch.from_numpy(image), -1, 0), int(box[0][0]),
                         int(box[0][1]), int(box[0][2]) - int(box[0][0]), int(box[0][3]) - int(box[0][1]), [224, 224]),
            -1, 0)

        # Normalization for input to dino trained transformer
        transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # Pass extracted images through dino  trained transformer
        embed = extractor.model(
            transform(torch.swapaxes(out, -1, 0).unsqueeze(0) / 255.0).type(torch.FloatTensor).to(device))

        org.append(org_image)
        embeds.append(np.array(embed.cpu()))
        seg.append(mask)
        print('Processed image {} from {}'.format(count, len(filenames)))
        count += 1

    print('Starting Kmeans clustering of extracted segments')
    kmean = KMeans(n_clusters=2)
    prediction = kmean.fit_predict(np.stack(embeds).squeeze(1))
    for i in range(len(seg)):
        seg[i] = util.apply_seg_map(org[i], seg[i] * (prediction[i] + 1),
                                    np.array(([0, 0, 255], [0, 255, 0], [255, 0, 0])), 0.4)

    util.im_show_n(seg, 5, 'co-segmentation')


if __name__ == '__main__':
    ################################################################################
    # GNN parameters
    ################################################################################
    # Numbers of epochs per step
    epochs = 10
    # Number of clusters
    K = 2
    ################################################################################
    # Processing parameters
    ################################################################################
    # Show only largest component in segmentation map (for k == 2)
    cc = True
    # Apply log binning to extracted descriptors (correspond to smoother segmentation maps)
    log_bin = False
    ################################################################################
    # Descriptors extraction parameters
    ################################################################################
    # Directory to pretrained Dino
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    # Resolution for dino input, higher res != better performance as Dino was trained on (224,224) sized images
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
    dir = './images/comb/'
    ################################################################################
    # Chose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If Dino Directory doesn't exist than download
    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)

    GNN_seg(epochs, K, pretrained_weights, dir, cc, log_bin, res, facet, layer, stride, device)
