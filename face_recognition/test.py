from PIL import Image
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

from tqdm import tqdm
import os

from data.Friends.dataset import FriendsDataset
import cv2
import random

import torch.backends.cudnn as cudnn

if __name__ == '__main__':

    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']

    trans = transforms.Compose([
        transforms.Resize([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    dataset_gallery = FriendsDataset(os.path.join(DATA_ROOT, 'Carpool_Gallery'))
    dataset_prove = FriendsDataset(os.path.join(DATA_ROOT, 'Carpool'), sorting=True)

    gallery_loader = torch.utils.data.DataLoader(dataset_gallery, batch_size = 1, pin_memory = True, num_workers = 0)
    prove_loader = torch.utils.data.DataLoader(dataset_prove, batch_size = 1, pin_memory = True, num_workers = 0)

    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                     'ResNet_101': ResNet_101(INPUT_SIZE), 
                     'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE), 
                     'IR_101': IR_101(INPUT_SIZE), 
                     'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                     'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                     'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        
    print("=" * 60)
    if os.path.isfile(BACKBONE_RESUME_ROOT):
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))    
    else:
        print("No Backbone Checkpoint Found at '{}'.".format(BACKBONE_RESUME_ROOT))
    print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    cudnn.benchmark = True
    
    #======= train & validation & save checkpoint =======#
    BACKBONE.eval()  # set to training mode

    Gallery_feature = torch.empty(0,512).to(DEVICE)
    Gallery_ID = []
    crop_size = 112
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
    for i, (data_path,img_path, txt_path) in enumerate(iter(dataset_gallery)):
        print('Processing %d/%d'%(i+1,len(dataset_gallery)))
        img = Image.open(img_path)
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        f=open(txt_path,'r')
        lines=f.readlines()
        f.close()

        if int(lines[1])>0:
            for i, line in enumerate(lines[2:]):
                b=line.split()
                facial5points = [[int(b[j]),int(b[j+1])] for j in [5,7,9,11,13]]
                warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
                img_warped = Image.fromarray(warped_face)
                if not os.path.isdir(data_path+'_aligned'):
                    os.mkdir(data_path+'_aligned')
                img_warped.save(os.path.join(data_path+'_aligned', img_path.rpartition('/')[-1].split('.')[0]+'_'+str(i)+'.png'))

                inputs = trans(img_warped).to(DEVICE).unsqueeze(0)
                features = BACKBONE(inputs)

                Gallery_ID.append(img_path.rpartition('/')[-1].split('.')[0])
                Gallery_feature=torch.cat([Gallery_feature, torch.div(features,torch.norm(features,2,1,True))],dim=0)

    print('Gallery_ID:',Gallery_ID)
    # Gallery_ID: ['Timothy Burke', 'Ross Geller', 'Joey Tribbiani', 'Monica Geller', 'Phoebe Buffay', 'Chandler Bing', 'Rachel Green']
    for i, (data_path,img_path, txt_path) in enumerate(iter(dataset_prove)):
        print('Processing %d/%d'%(i+1,len(dataset_prove)))
        if int(img_path.rpartition('/')[-1].split('.')[0].rpartition('_')[2]) < 660:
            continue
        img = Image.open(img_path)
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        f=open(txt_path,'r')
        lines=f.readlines()
        f.close()

        if not os.path.isdir(data_path+'_result'):
            os.mkdir(data_path+'_result')
        
        if int(lines[1])>0:
            f=open(os.path.join(data_path+'_result',img_path.rpartition('/')[-1].split('.')[0]+'.txt'),'w')
            for i, line in enumerate(lines[2:]):
                b=line.split()
                facial5points = [[int(b[j]),int(b[j+1])] for j in [5,7,9,11,13]]
                warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
                img_warped = Image.fromarray(warped_face)
                if not os.path.isdir(data_path+'_aligned'):
                    os.mkdir(data_path+'_aligned')
                img_warped.save(os.path.join(data_path+'_aligned', img_path.rpartition('/')[-1].split('.')[0]+'_'+str(i)+'.png'))

                inputs = trans(img_warped).to(DEVICE).unsqueeze(0)
                features = BACKBONE(inputs)

                similarity_matrix = Gallery_feature.matmul(torch.div(features,torch.norm(features,2,1,True)).transpose(0,1))
                similarity = similarity_matrix.max().item()
                identity = similarity_matrix.argmax().item()
                f.write(line.rstrip('\n') + ' ' + str(similarity) + ' ' + str(identity) + '\n')
            f.close()

    


    color_dict = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(Gallery_ID))]
    for i, (data_path,img_path, txt_path) in enumerate(iter(dataset_prove)):
        print('Processing %d/%d'%(i+1,len(dataset_prove)))
        if int(img_path.rpartition('/')[-1].split('.')[0].rpartition('_')[2]) < 660:
            continue
        if not os.path.exists(os.path.join(data_path+'_result',img_path.rpartition('/')[-1].split('.')[0]+'.txt')):
            if not os.path.isdir(data_path+'_video'):
                os.mkdir(data_path+'_video')
            #img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
            #cv2.imwrite(os.path.join(data_path+'_video',img_path.rpartition('/')[-1]), img_raw)
            continue
        f=open(os.path.join(data_path+'_result',img_path.rpartition('/')[-1].split('.')[0]+'.txt'),'r')
        lines=f.readlines()
        f.close()
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)

        for line in lines:
            b=line.split()            
            #if int(b[2]) < 40 or int(b[3]) < 40 or float(b[-2]) < 0.5:
            if float(b[-2]) < 0.2:
                continue
            
            cv2.rectangle(img_raw, (int(b[0]), int(b[1])), (int(b[0])+int(b[2]), int(b[1])+int(b[3])), color_dict[int(b[-1])], 2, lineType=cv2.LINE_AA)
            t_size = cv2.getTextSize(Gallery_ID[int(b[-1])], 0, fontScale=2/3, thickness=2)[0]
            c1 = int(b[0]), int(b[1])
            c2 = int(b[0]) + t_size[0], int(b[1]) - t_size[1] - 3
            cv2.rectangle(img_raw, c1, c2, color_dict[int(b[-1])], -1, cv2.LINE_AA)  # filled
            #cv2.putText(img_raw, Gallery_ID[int(b[-1])] + ' ' + str(b[-2]), (c1[0], c1[1] - 2), 0, 2/3, [225, 255, 255])
            cv2.putText(img_raw, Gallery_ID[int(b[-1])], (c1[0], c1[1] - 2), 0, 2/3, [225, 255, 255])
