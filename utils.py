import torch
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import time
from easydict import EasyDict as edict
import yaml
from PIL import Image
import imageio

def read_data_file(filename, root=None):
    
    lists = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        while line:
            info = line.strip().split(' ')
            if root is not None:
                info[0] = os.path.join(root, info[0])
            if len(info) == 1:
                item = (info[0], 1)
            else:
                item = (info[0], int(info[1]))
            lists.append(item)
            line = fp.readline()
    
    return lists

def pil_loader(path):
    
    img = Image.open(path)
    return img.convert('RGB')

class MydataFolder(data.Dataset):
    """A data loader where the list is arranged in this way:
        
        dog/1.jpg 1
        dog/2.jpg 1
        dog/3.jpg 1
            .
            .
            .
        cat/1.jpg 2
        cat/2.jpg 2
            .
            .
            .
        path      label
    
    Args:
      
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
    """

    def __init__(self, filename, root=None, transform=None):

        lists = read_data_file(filename)
        if len(lists) == 0:
            raise(RuntimeError('Found 0 images in subfolders\n'))

        self.root = root
        self.transform = transform
        self.lists = lists
        self.load = pil_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): index
    
        Returns:
            tuple: (image, label) where label is the clas of the image
        """
    
        path, label = self.lists[index]
        img = self.load(path)
    
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        
        return len(self.lists)

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

def plot_loss(d_loss, g_loss, num_epoch, epoches, save_dir):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0,epoches + 1)
    ax.set_ylim(0, max(np.max(g_loss), np.max(d_loss)) * 1.1)
    plt.xlabel('Epoch {}'.format(num_epoch))
    plt.ylabel('Loss')
    
    plt.plot([i for i in range(1, num_epoch + 1)], d_loss, label='Discriminator', color='red', linewidth=3)
    plt.plot([i for i in range(1, num_epoch + 1)], g_loss, label='Generator', color='mediumblue', linewidth=3)
    
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(num_epoch)))
    plt.close()

def plot_result(G, fixed_noise, image_size, num_epoch, save_dir, fig_size=(8, 8), is_gray=False):

    G.eval()
    generate_images = G(fixed_noise)
    G.train()
    
    n_rows = n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    
    for ax, img in zip(axes.flatten(), generate_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if is_gray:
            img = img.cpu().data.view(image_size, image_size).numpy()
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
            ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, title, ha='center')
    
    plt.savefig(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(num_epoch)))
    plt.close()

def create_gif(epoches, save_dir):
    
    images = []
    for i in range(1, epoches + 1):
        images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_epoch_{}.png'.format(i))))
    imageio.mimsave(os.path.join(save_dir, 'result.gif'), images, fps=5)
    
    images = []
    for i in range(1, epoches + 1):
        images.append(imageio.imread(os.path.join(save_dir, 'DCGAN_loss_epoch_{}.png'.format(i))))
    imageio.mimsave(os.path.join(save_dir, 'result_loss.gif'), images, fps=5)

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
