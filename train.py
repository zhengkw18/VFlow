"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import Glow
from tqdm import tqdm


def main(args):
    global img_shape
    # Set up main device and scale batch size
    # device = ('cuda:' + str(args.gpu_ids[0])) if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    torch.cuda.set_device(args.gpu_ids[0])
    args.batch_size *= max(1, len(args.gpu_ids))
    args.lr *= args.batch_size / 64
    args.warm_up /= args.batch_size / 64
    args.decay_step /= args.batch_size / 64

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # No normalization applied, since Glow expects inputs in (0, 1)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    img_shape = (args.batch_size, 3, 32, 32)
    # Model
    print('Building model..')
    net = Glow(img_shape=img_shape, num_channels=args.num_channels, num_levels=args.num_levels, num_steps=args.num_steps)
    net = torch.nn.DataParallel(net.cuda(), device_ids=args.gpu_ids)
    cudnn.benchmark = args.benchmark
    print("Using", len(args.gpu_ids), "GPUs")

    start_epoch = 0
    if args.resume and os.path.isdir('ckpts'):
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        global training_step
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch'] + 1
        training_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler1 = sched.LambdaLR(optimizer, lambda s: min(min(1., s / args.warm_up), max(args.decay_rate**(s - args.decay_step), 0.25)))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, optimizer, scheduler1, loss_fn, args.max_grad_norm)
        test(epoch, net, testloader, loss_fn, args.num_samples)


@torch.enable_grad()
def train(epoch, net, trainloader, optimizer, scheduler1, loss_fn, max_grad_norm):
    global training_step
    global img_shape
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.cuda(non_blocking=True)
            optimizer.zero_grad()
            _, sldj = net(x, reverse=False)
            loss = loss_fn(img_shape, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler1.step(training_step)
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(img_shape, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            training_step += 1


@torch.no_grad()
def sample(net, batch_size, testloader):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    for x, _ in testloader:
        x = x.cuda(non_blocking=True)
        x, _ = net(x, reverse=True)
        x = x[0:batch_size, :, :, :]
        break
    x = torch.sigmoid(x)

    return x


@torch.no_grad()
def test(epoch, net, testloader, loss_fn, num_samples):
    global best_loss
    global img_shape
    net.eval()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.cuda(non_blocking=True)
            _, sldj = net(x, reverse=False)
            loss = loss_fn(img_shape, sldj)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(img_shape, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, num_samples, testloader)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VFlow on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=192, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1.2e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=96, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=10, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=True, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=2000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--decay_rate', default=0.999, type=int, help='Decay rate for lr')
    parser.add_argument('--decay_step', default=3000, type=int, help='Number of steps before lr decay')

    best_loss = 1e8
    training_step = 0
    img_shape = (0, 0, 0, 0)
    main(parser.parse_args())
