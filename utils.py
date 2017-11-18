import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
from easydict import EasyDict as edict
import yaml

class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_result(G, fixed_noise, image_size, num_epoch, save_dir, fig_size=(8, 8)):

    G.eval()
    generate_images = G(fixed_noise)
    G.train()

    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        img = img.cpu().data.view(image_size, image_size).numpy()
        ax.imshow(img, cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')

    plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(num_epoch)))
    plt.close()

def save_checkpoint(state, filename='checkpoint'):

    torch.save(state, filename + '.pth.tar')

def Config(filename):

    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print '{}: {}'.format(x, parser[x])
    print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())

    return parser

def print_log(epoch, epoches, iteration, iters, learning_rate,
              display, batch_time, data_time, D_losses, G_losses):
    print('epoch: [{}/{}] iteration: [{}/{}]\t'
        'Learning rate: {}').format(epoch, epoches, iteration, iters, learning_rate)
    print('Time {batch_time.sum:.3f}s / {0}iters, ({batch_time.avg:.3f})\t'
      'Data load {data_time.sum:.3f}s / {0}iters, ({data_time.avg:3f})\n'
      'Loss_D = {loss_D.val:.8f} (ave = {loss_D.avg:.8f})\n'
      'Loss_G = {loss_G.val:.8f} (ave = {loss_G.avg:.8f})\n'.format(
      display, batch_time=batch_time,
      data_time=data_time, loss_D=D_losses, loss_G=G_losses))
    print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())
