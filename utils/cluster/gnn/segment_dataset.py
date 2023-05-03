from torchvision.utils import draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from mat_extract import descriptor_mat
from torch_geometric.data import Data
from extractor import ViTExtractor
import matplotlib.pyplot as plt
from gnn_pool import GNNpool
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import torch
import util
import cv2
import os


def GNN_seg(epoch, K, pretrained_weights, dir, out_dir, cc, b_box, log_bin, res, facet, layer, stride, device):
    """
    Segment entire dataset; Get bounding box (k==2 only) or segmentation maps
    bounding boxes will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')
    @param epoch: Number of epochs for every step in image
    @param K: Number of segments to search in each image
    @param pretrained_weights: Weights of pretrained images
    @param dir: Directory for chosen dataset
    @param out_dir: Output directory to save results
    @param cc: If k==2 chose the biggest component, and discard the rest (only available for k==2)
    @param b_box: If true will output bounding box (for k==2 only), else segmentation map
    @param log_bin: Apply log binning to the descriptors (correspond to smother image)
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

    for filename in tqdm(os.listdir(dir)):
        # If not image, skip
        if not filename.endswith((".jpg", ".png")):
            continue

        # loading images
        image_tensor, org_image = util.load_data_img(dir + filename, res)
        # Extract deep features, and create an adj matrix
        W, F, D = descriptor_mat(image_tensor, extractor, layer, facet, bin=log_bin, device=device)

        adj = W
        feats = F
        image = org_image

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(adj, feats)
        data = Data(node_feats, edge_index, edge_weight).to(device)

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

        # Apply bounding box to the largest component (for k == 2)
        if b_box:
            box = masks_to_boxes(torch.from_numpy(mask).unsqueeze(0))
            image = draw_bounding_boxes(torch.from_numpy(np.einsum('ijk->kij', image)), box, colors='red').numpy()
            cord = list(box.numpy().astype(np.int64)[0])

            # saving b_box cords format: class, confidence, left, top , right, bottom
            with open(out_dir + filename.split('.')[0] + '.txt', 'w+') as f:
                f.write('1 ')
                f.write('1 ')
                length = len(cord)
                for i in range(length):
                    f.write(str(cord[i]))
                    if i < length - 1:
                        f.write(' ')
                f.close()
        else:
            plt.imsave(out_dir + filename, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GNN', add_help=False)
    ################################################################################
    # GNN parameters
    ################################################################################
    parser.add_argument('--epochs', default=10, type=int, help="""Number of epochs per image""")
    parser.add_argument('--K', default=2, type=int, help="""Number of clusters to find for each image""")
    ################################################################################
    # Processing parameters
    ################################################################################
    parser.add_argument('--b_box', default=False, type=bool,
                        help="""If true output of the model will be a bounding box coordinates of main object; else a segmentation map
        Only available while K == 2""")
    parser.add_argument('--cc', default=False, type=bool,
                        help="""If true output will be the largest connected component of segmentation map;
        Only available while K == 2""")
    parser.add_argument('--log_bin', default=False, type=bool,
                        help="""Log binning of descriptors""")
    ################################################################################
    # Descriptors extraction parameters
    ################################################################################
    # Directory to pretrained Dino
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    parser.add_argument('--res', default=(224, 224), type=tuple, help="""Resolution for dino input, higher res != 
    better performance as Dino was trained on (224,224) sized images""")
    parser.add_argument('--stride', default=4, type=int, help="""stride for descriptor extraction""")
    parser.add_argument('--facet', default='key', type=str,
                        help="""facet for descriptor extraction (key/query/value)""")
    parser.add_argument('--layer', default=11, type=int, help="""layer to extract descriptors from""")
    ################################################################################
    # Data parameters
    ################################################################################
    parser.add_argument('--in_dir', default='./images/cat/', type=str, help="""input dir for images to segment""")
    parser.add_argument('--out_dir', default='./results/', type=str, help="""output dir for segmented images""")
    ################################################################################

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()

    # Check for mistakes in given arguments
    if args.K != 2 and (args.b_box or args.cc):
        print('Connected components and bounding boxes only available for k == 2')
        exit()

    # If Directory doesn't exist than download
    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)

    GNN_seg(args.epochs, args.K, pretrained_weights, args.in_dir, args.out_dir, args.cc, args.b_box, args.log_bin,
            args.res, args.facet, args.layer, args.stride, device)
