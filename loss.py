import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16



def reconstruction_loss(pred, target):
    return nn.L1Loss()(pred, target)

def consistency_loss(t_map, l_map):
    t_map_norm = (t_map - t_map.min()) / (t_map.max() - t_map.min() + 1e-8)
    l_map_norm = (l_map - l_map.min()) / (l_map.max() - l_map.min() + 1e-8)
    return nn.L1Loss()(t_map_norm, l_map_norm)

def perceptual_loss(vgg, pred, target):
    pred_features = vgg(pred)
    target_features = vgg(target)
    return nn.L1Loss()(pred_features, target_features)

def hazy_reconstruction_loss(hazy, dehazed, t_map, a_map):
    reconstructed_hazy = dehazed * t_map + a_map * (1 - t_map)
    return nn.L1Loss()(hazy, reconstructed_hazy)

def total_variation_loss(image):
    batch_size, channels, height, width = image.size()
    tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return tv_h + tv_w
