import os
# Using this code to force the usage of any specific GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torch.utils.data as data
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from math import log10
import torchvision
import cv2
import skimage
import scipy.io
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from model import losses
from model.networks import *
from util.model_storage import save_checkpoint
from data.dataloader import *

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", default="./pretrained/weight.pth", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--batch_size", default="8", type=int, help="The path to store our batch_size")
parser.add_argument("--image_dir", default="./data/test_img/", type=str, help="The path to store our batch_size")
parser.add_argument("--image_list", default="./data/test_fileList.txt", type=str, help="The path to store our batch_size")

global opt,model
opt = parser.parse_args()

fsrnet = define_G(input_nc = 3, output_nc = 3, ngf=64, which_model_netG=0)

if torch.cuda.is_available():
    fsrnet = fsrnet.cuda()

if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)

        pretrained_dict = weights['model'].state_dict()
        model_dict = fsrnet.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        fsrnet.load_state_dict(model_dict)
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

demo_dataset = TestDatasetFromFile(
    opt.image_list,
    opt.image_dir)  
test_data_loader = data.DataLoader(dataset=demo_dataset, batch_size=opt.batch_size, num_workers=8, drop_last=True,
                                    pin_memory=True)

for iteration, batch in enumerate(test_data_loader):
    input = Variable(batch[0])
    input = input.cuda()
    upscaled,boundaries,reconstructed = fsrnet(input)

    if not os.path.isdir('./test_result/Coarse_SR_network'):
        os.makedirs('./test_result/Coarse_SR_network')
    if not os.path.isdir('./test_result/Prior_Estimation'):
        os.makedirs('./test_result/Prior_Estimation')
    if not os.path.isdir('./test_result/Final_SR_reconstruction'):
        os.makedirs('./test_result/Final_SR_reconstruction')

    for index in range(opt.batch_size):
        final_output = reconstructed.permute(0,2,3,1).detach().cpu().numpy()
        final_output_0 = final_output[index,:,:,:]

        estimated_boundary = boundaries.permute(0,2,3,1).detach().cpu().numpy()
        estimated_boundary_0 = estimated_boundary[index,:,:,0]

        output = upscaled.permute(0,2,3,1).detach().cpu().numpy()
        output_0 = output[index,:,:,:]

        img_num = iteration*opt.batch_size + index

        scipy.misc.toimage(output_0 * 255, high=255, low=0, cmin=0, cmax=255).save(
                './test_result/Coarse_SR_network/%4d.jpg'% (img_num))
        scipy.misc.toimage(estimated_boundary_0 * 255, high=255, low=0, cmin=0, cmax=255).save(
                './test_result/Prior_Estimation/%4d.jpg' % (img_num))
        scipy.misc.toimage(final_output_0 * 255, high=255, low=0, cmin=0, cmax=255).save(
                './test_result/Final_SR_reconstruction/%4d.jpg' % (img_num))
        #code minor changeA2

