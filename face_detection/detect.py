#!/usr/bin/env python
import argparse
import os
import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from data import cfg_mnet, cfg_re50
from model.prior_box import PriorBox
from model.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/resnet50_final.pt',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobilenet0.25', choices={"mobilenet0.25", "resnet50"})
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence-threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top-k', default=5000, type=int, help='top_k')
parser.add_argument('--nms-threshold', default=0.4, type=float, help='NMS threshold')
parser.add_argument('--keep-top-k', default=750, type=int, help='keep top k')
parser.add_argument('-s', '--save-image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis-thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict["net_state_dict"], strict=False)
    return model


@torch.no_grad()
def main():
    cfg = None
    if args.network == "mobilenet0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    for i in range(100):
        image_path = "./curve/test.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize

        # ignore low scores
        inds = torch.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()
        boxes = boxes[order][:args.top_k]
        landms = landms[order][:args.top_k]
        scores = scores[order][:args.top_k]

        # do NMS
        keep = nms(boxes, scores, args.nms_threshold)

        boxes = boxes[keep]
        scores = scores[keep]
        landms = landms[keep]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        landms = landms.cpu().numpy()
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            cv2.imwrite(name, img_raw)


if __name__ == "__main__":
    main()