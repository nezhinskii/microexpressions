import os, cv2
import numpy as np
from PIL import Image, ImageFilter
import logging
import torch
import torch.nn as nn
import random
import time
from scipy.integrate import simpson

def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]
        
    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:]
        dists = np.sum(np.power(pt-meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1+num_nb])
    
    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)
    
    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len
    
    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

def forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb):
    net.eval()
    with torch.no_grad():
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()

        # Reshape outputs while preserving batch dimension
        outputs_cls = outputs_cls.view(tmp_batch, tmp_channel, -1)  # [batch_size, num_lms, height*width]
        max_ids = torch.argmax(outputs_cls, dim=2)  # [batch_size, num_lms]
        max_cls = torch.max(outputs_cls, dim=2)[0]  # [batch_size, num_lms]

        # Select offsets for maximum indices
        outputs_x = outputs_x.view(tmp_batch, tmp_channel, -1)  # [batch_size, num_lms, height*width]
        outputs_y = outputs_y.view(tmp_batch, tmp_channel, -1)  # [batch_size, num_lms, height*width]
        max_ids_expanded = max_ids.unsqueeze(2)  # [batch_size, num_lms, 1]
        outputs_x_select = torch.gather(outputs_x, 2, max_ids_expanded).squeeze(2)  # [batch_size, num_lms]
        outputs_y_select = torch.gather(outputs_y, 2, max_ids_expanded).squeeze(2)  # [batch_size, num_lms]

        # Neighbor offsets
        max_ids_nb = max_ids.unsqueeze(2).repeat(1, 1, num_nb)  # [batch_size, num_lms, num_nb]
        outputs_nb_x = outputs_nb_x.view(tmp_batch, tmp_channel*num_nb, -1)  # [batch_size, num_lms*num_nb, height*width]
        outputs_nb_y = outputs_nb_y.view(tmp_batch, tmp_channel*num_nb, -1)  # [batch_size, num_lms*num_nb, height*width]
        outputs_nb_x_select = torch.gather(outputs_nb_x, 2, max_ids_nb.view(tmp_batch, -1, 1)).view(tmp_batch, tmp_channel, num_nb)  # [batch_size, num_lms, num_nb]
        outputs_nb_y_select = torch.gather(outputs_nb_y, 2, max_ids_nb.view(tmp_batch, -1, 1)).view(tmp_batch, tmp_channel, num_nb)  # [batch_size, num_lms, num_nb]

        # Compute coordinates
        tmp_x = (max_ids % tmp_width).float() + outputs_x_select  # [batch_size, num_lms]
        tmp_y = (max_ids // tmp_width).float() + outputs_y_select  # [batch_size, num_lms]
        tmp_x /= 1.0 * input_size / net_stride  # Normalize to [0, 1]
        tmp_y /= 1.0 * input_size / net_stride

        tmp_nb_x = (max_ids % tmp_width).float().unsqueeze(2) + outputs_nb_x_select  # [batch_size, num_lms, num_nb]
        tmp_nb_y = (max_ids // tmp_width).float().unsqueeze(2) + outputs_nb_y_select  # [batch_size, num_lms, num_nb]
        tmp_nb_x /= 1.0 * input_size / net_stride
        tmp_nb_y /= 1.0 * input_size / net_stride

    return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls