import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from network import *
from utils import *

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='the path of save generate images')

    return parser.parse_args()

def construct_model(args):

    G = generator(z_size=config.z_size, out_size=config.out_size, ngf=config.ngf).cuda()
    print 'G network structure'
    print G
    D = discriminator(in_size=config.in_size, ndf=config.ndf).cuda()
    print 'D network structure'
    print D
    return G, D

def train_net(G, D, args, config):

    cudnn.benchmark = True
    traindir = args.train_dir

    if config.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(traindir, True,
                    transforms.Compose([transforms.Scale(config.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]), download=True),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    elif config.dataset == 'celebA':
        return
    else:
        return

    # setup loss function
    criterion = nn.BCELoss().cuda()

    # setup optimizer
    optimizerD = torch.optim.Adam(D.parameters(), lr=config.base_lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=config.base_lr, betas=(0.5, 0.999))

    # setup some varibles
    batch_time = AverageMeter()
    data_time = AverageMeter()
    D_losses = AverageMeter()
    G_losses = AverageMeter()

    fixed_noise = torch.FloatTensor(8 * 8, config.z_size, 1, 1).normal_(0, 1)
    fixed_noise = Variable(fixed_noise.cuda(), volatile=True)

    end = time.time()

    D.train()
    G.train()

    for epoch in range(config.epoches):
        for i, (input, _) in enumerate(train_loader):
            '''
                Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            '''
            data_time.update(time.time() - end)

            batch_size = input.size(0)
            input_var = Variable(input.cuda())

            # Train discriminator with real data
            label_real = torch.ones(batch_size)
            label_real_var = Variable(label_real.cuda())

            D_real_result = D(input_var).squeeze()
            D_real_loss = criterion(D_real_result, label_real_var)

            # Train discriminator with fake data
            label_fake = torch.zeros(batch_size)
            label_fake_var = Variable(label_fake.cuda())

            noise = torch.randn((batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).squeeze()
            D_fake_loss = criterion(D_fake_result, label_fake_var)

            # Back propagation
            D_train_loss = D_real_loss + D_fake_loss
            D_losses.update(D_train_loss.data[0])

            D.zero_grad()
            D_train_loss.backward()
            optimizerD.step()

            '''
                Update G network: maximize log(D(G(z)))
            '''
            noise = torch.randn((batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).squeeze()
            G_train_loss = criterion(D_fake_result, label_real_var)
            G_losses.update(G_train_loss.data[0])

            # Back propagation
            D.zero_grad()
            G.zero_grad()
            G_train_loss.backward()
            optimizerG.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.display == 0 or (i + 1) == len(train_loader):
                print_log(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr,
                          config.display, batch_time, data_time, D_losses, G_losses)
                batch_time.reset()
                data_time.reset()

        D_losses.reset()
        G_losses.reset()

        # plt the generate images
        plot_result(G, fixed_noise, config.image_size, epoch + 1, args.save_dir)

        # save the D and G.
        save_checkpoint({'epoch': epoch, 'state_dict': D.state_dict(),}, os.path.join(args.save_dir, 'D_epoch_{}'.format(epoch)))
        save_checkpoint({'epoch': epoch, 'state_dict': G.state_dict(),}, os.path.join(args.save_dir, 'G_epoch_{}'.format(epoch)))

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parse()
    config = Config(args.config)
    G, D = construct_model(config)
    train_net(G, D, args, config)
