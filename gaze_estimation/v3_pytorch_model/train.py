import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import os
import time

from ir_data import IR_FACE_Dataset
from config import Config
import numpy as np

from torch.utils.data import Dataset, DataLoader

from utils import AverageMeter
# ----------------------------------
if Config.use_model_type == 'LIGHT':
    from gaze_model_light_ver import Estimator
elif Config.use_model_type == 'HEAVY' or Config.use_model_type == 'HEAVY+ATT':
    from gaze_model_heavy_ver import Estimator

# ----------------------------------

def train():
    torch.multiprocessing.freeze_support()
    train_transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ])

    ir_dataset = IR_FACE_Dataset(data_path=Config.data_path, \
        img_w=Config.global_img_size[0] ,img_h=Config.global_img_size[1], img_local_h=Config.local_img_size[1], \
        transform=train_transform)
    ir_dataloader = DataLoader(ir_dataset, batch_size=Config.batch_size, \
        shuffle=True, num_workers=1)

    device = torch.device("cuda")

    # checkpt dir
    if os.path.exists(Config.save_path) == False:
        os.makedirs(Config.save_path)

    # model
    if Config.use_model_type == 'HEAVY+ATT':
        model = Estimator(use_attention_map=True).cuda()
    else:
        model = Estimator().cuda()
    model = model.to(device)

    # opt
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr, momentum=Config.momentum, \
        weight_decay=Config.weight_decay)


    for epoch_i in range(Config.max_epoch):
        model.train()

        #Config.lr = adjust_learning_rate_v2(optimizer, epoch_i - 1, Config)
        #for param_group in optimizer.param_groups:
        #    param_group["lr"] = Config.lr

        iter_max = ir_dataset.__len__() // Config.batch_size

        # for print
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        
        dataiter = iter(ir_dataloader)
        steps_per_epoch = iter_max + 1
        #for ii, data in enumerate(ir_dataloader):
        for ii in range(steps_per_epoch):

            data_time.update(time.time() - end)

            data_input, data_input_local, label = dataiter.next()
            data_input = data_input.to(device)
            targets = label.to(device)
            data_input_local = data_input_local.to(device)
            
            

            # optimizer step
            optimizer.zero_grad()
            outputs = model(data_input, data_input_local)
            loss = criterion(outputs, torch.argmax(targets, 1))

            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            total = data_input.size(0)
            _, predicted = outputs.max(1)
            correct = predicted.eq(torch.argmax(targets,1)).sum().item()
            top1.update(100.*correct/total)

            losses.update(loss.item(), data_input.size(0))


            end = time.time()

            if ii % Config.print_iter == 0:
                print('\nEpoch: [%d | %d], Iter : [%d | %d] LR: %f | Loss : %f | top1 : %.4f | batch_time : %.3f'  \
                    % (epoch_i, Config.max_epoch, ii, iter_max + 1, Config.lr, losses.avg, top1.avg, data_time.val))


             # measure elapsed time


        # save model
        if epoch_i % Config.save_epoch == 0:
            torch.save({'state_dict' : model.state_dict(), 'opt' : optimizer.state_dict()}, \
                Config.save_path + "/check_" + str(epoch_i) + ".pth")
        

    
# not using -
def adjust_learning_rate(optimizer, epoch, config):
    global state
    if epoch in config.schedule:
        config.lr *= config.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.lr

def adjust_learning_rate_v2(optimizer, epoch, config):
    lr = config.lr * (0.1 ** (epoch // 10))
    return lr

if __name__ == '__main__':
    train()