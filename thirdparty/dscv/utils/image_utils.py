import numpy as np
import torch
import torchvision


def pil_img_type_to(img, type_to, **kwargs):
    if type_to == np.ndarray:
        return np.array(img, **kwargs)
    else:
        raise NotImplementedError('')


def tensor_img_type_to(img, type_to, **kwargs):
    '''
    Args:
        img     : (tensor) has shape of (C, H, W)
    '''
    if type_to == np.ndarray:
        # np.ndarray img has shape of (H, W, C)
        return np.asarray(img.permute(1, 2, 0).cpu().detach(), **kwargs)
    else:
        raise NotImplementedError('')


def img_type_converter(img, type_to, **kwargs):
    if torch.is_tensor(img):
        return tensor_img_type_to(img, type_to, **kwargs)
    else:
        raise NotImplementedError('')


def denormalize(img, mean, std):
    _denormalize = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0., 0., 0.], 1 / std),
        torchvision.transforms.Normalize(-mean, [1., 1., 1.])
    ])
    return _denormalize(img)
