#!/usr/bin/env python
import argparse
import datetime
import math
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import (WiderFaceDetection, cfg_mnet, cfg_re50, detection_collate,
                  preproc)
from model.prior_box import PriorBox
from model.multibox_loss import MultiBoxLoss
from model.retinaface import RetinaFace
from utils.utils import load_model

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobilenet0.25', choices={"mobilenet0.25", "resnet50"})
parser.add_argument('--batch-size', default=32, help='Batch size')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume-net', default=None, help='resume net for retraining')
parser.add_argument('--resume-epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save-folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()


os.makedirs(args.save_folder, exist_ok=True)
if args.network == "mobilenet0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.dataset
save_folder = args.save_folder


def initialize_network():
    net = RetinaFace(**cfg)
    print("Printing net...")
    print(net)

    if args.resume_net is not None and os.path.isfile(args.resume_net):
        print('Loading resume network...')
        net = load_model(net, args.resume_net, True, argsis_train=True)

    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    return net


def train(net, optimizer, criterion, dataloader):
    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    epoch_size = math.ceil(len(dataloader))
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        load_t0 = time.perf_counter()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except:
            batch_iterator = iter(dataloader)
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
                torch.save(
                    {
                        "net_state_dict": net_state_dict,
                        "epoch": epoch,
                    }, save_folder + f"{cfg['backbone']}_epoch{epoch:03d}.pt"
                )
            epoch += 1
            images, targets = next(batch_iterator)

        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad(set_to_none=True)
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()

        load_t1 = time.perf_counter()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print(f"Epoch:{epoch:03d}/{max_epoch:03d} "
              '|| Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format((iteration % epoch_size) + 1,
              epoch_size, iteration+1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    net_state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(
        {
            "net_state_dict": net_state_dict,
            "epoch": epoch,
        }, save_folder + f"{cfg['backbone']}_final.pt"
    )


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    net = initialize_network()
    torch.backends.cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    dataloader = DataLoader(
        dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate
    )

    train(net, optimizer, criterion, dataloader)


if __name__ == '__main__':
    main()
