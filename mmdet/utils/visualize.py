import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

dirname = '/ws/data/dshong/imgs'

def imsave(image, title=None, save=None):
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    if save:
        plt.savefig(f"{dirname}/{save}.png")

def bincount(data, num_bins):
    min_, max_ = data.min(), data.max()
    # x = torch.linspace(min_, max_, steps=num_bins)
    # return data.bincount(x)
    # return torch.bincount(data, minlength=2)
    return torch.histc(data, bins=num_bins, min=float(min_), max=float(max_))

def multi_imsave(image, rows, cols, save=None):
    plt.figure(figsize=(14, 10))
    i = 0
    save_, title_ = None, None
    for row in range(rows):
        for col in range(cols):
            if (row == rows-1) and (col == cols-1):
                save_ = save
            plt.subplot(rows, cols, i+1)
            count = bincount(image[i].reshape(-1), 2)
            torch.set_printoptions(precision=3, sci_mode=False)
            imsave(image[i].cpu().detach().numpy(), f"{count}", save_)
            torch.set_printoptions(precision=None, sci_mode=True)
            i = i + 1
