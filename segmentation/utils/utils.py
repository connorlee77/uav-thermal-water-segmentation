import numpy as np
from skimage.morphology import binary_closing
import cv2
import torch

def merge_pseudolabels(nn_mask, sift_pseudolabel=None, flow_pseudolabel=None, sky_pseudolabel=None, cpu=False):

    if cpu:
        nn_mask = torch.from_numpy(nn_mask).unsqueeze(0)

        if sift_pseudolabel is not None:
            sift_pseudolabel = torch.from_numpy(sift_pseudolabel).unsqueeze(0)
        if flow_pseudolabel is not None:
            flow_pseudolabel = torch.from_numpy(flow_pseudolabel).unsqueeze(0)
        if sky_pseudolabel is not None:
            sky_pseudolabel = torch.from_numpy(sky_pseudolabel).unsqueeze(0)
    
    sigmoid_mask = torch.sigmoid(nn_mask)
    final_river_label = [sigmoid_mask[:,1]]
    final_shore_label = [sigmoid_mask[:,0]]
    
    device = sigmoid_mask.device

    B = sigmoid_mask.shape[0]
    river_weights_sum = torch.ones(B).view(B, 1, 1).to(device)
    shore_weights_sum = torch.ones(B).view(B, 1, 1).to(device)

    valid_mask = torch.ones_like(sigmoid_mask)
    if sift_pseudolabel is not None:
        sift_shore_prob = 1 - sift_pseudolabel
        sift_river_prob = sift_pseudolabel

        river_weight = torch.ones_like(river_weights_sum)
        shore_weight = torch.ones_like(river_weights_sum)

        final_river_label.append(river_weight*sift_river_prob)
        final_shore_label.append(shore_weight*sift_shore_prob)

        river_weights_sum += river_weight
        shore_weights_sum += shore_weight

    if flow_pseudolabel is not None:
        flow_shore_prob = 1 - flow_pseudolabel
        flow_river_prob = flow_pseudolabel

        river_weight = torch.ones_like(river_weights_sum) 
        shore_weight = torch.zeros_like(river_weights_sum)

        bad_batches = torch.sum(flow_pseudolabel.view(B, -1) == -1, dim=1) > 0
        river_weight[bad_batches] = 0
        shore_weight[bad_batches] = 0

        final_river_label.append(river_weight*flow_river_prob)
        final_shore_label.append(shore_weight*flow_shore_prob)

        river_weights_sum += river_weight
        shore_weights_sum += shore_weight

    merged_river_label = torch.stack(final_river_label, dim=1).sum(dim=1, keepdim=True) / river_weights_sum.view(B, 1, 1, 1)
    merged_shore_label = torch.stack(final_shore_label, dim=1).sum(dim=1, keepdim=True) / shore_weights_sum.view(B, 1, 1, 1)

    if sky_pseudolabel is not None:
        merged_river_label[sky_pseudolabel.unsqueeze(1)] = 0.05
        merged_shore_label[sky_pseudolabel.unsqueeze(1)] = 0.95

    final_label = torch.cat([merged_shore_label, merged_river_label], dim=1)

    valid_mask_mask = torch.logical_and(final_label > 0.4, final_label < 0.6)
    valid_mask[valid_mask_mask] = 0

    if cpu:
        return final_label.squeeze().numpy(), valid_mask.squeeze().numpy()

    return final_label, valid_mask    

def momentum_update(network1, network2, momentum_rate=0.3):
    # Update weights of network 1 with network 2
    for param_1, param_2 in zip(network1.parameters(), network2.parameters()):
        param_1.data = param_1.data * (1 - momentum_rate) + param_2.data * momentum_rate

def postprocess_mask(mask):
    # Keep biggest water mass
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        
    mask = mask / 255
    mask = binary_closing(mask.astype(bool))
    return 255*np.uint8(mask)