#!/usr/bin/env python
import argparse
import os
import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from model.prior_box import PriorBox
from model.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.misc import draw_keypoint

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    default='./weights/mobilenet0.25_final.pt',
    help='Trained state_dict file path to open'
)
parser.add_argument(
    '--image',
    help='Input image file to detect'
)
parser.add_argument(
    '--cpu', action="store_true", default=False,
    help='Use cpu inference'
)
parser.add_argument(
    '--confidence-threshold', type=float, default=0.02,
    help='confidence_threshold'
)
parser.add_argument(
    '--top-k', type=int, default=5000,
    help='top_k'
)
parser.add_argument(
    '--nms-threshold', type=float, default=0.4,
    help='NMS threshold'
)
parser.add_argument(
    '--keep-top-k', type=int, default=750,
    help='keep top k'
)
parser.add_argument(
    '-s', '--save-image', action="store_true", default=False,
    help='show detection results'
)
parser.add_argument(
    '--vis-thres', type=float, default=0.6,
    help='visualization_threshold'
)


@torch.no_grad()
def main():
    args = parser.parse_args()
    assert os.path.isfile(args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    device = torch.device("cpu" if args.cpu else "cuda")

    # net and model
    net = RetinaFace(**cfg)
    net.load_state_dict(checkpoint["net_state_dict"], strict=False)
    net.eval().requires_grad_(False)
    net.to(device)
    print('Finished loading model!')

    resize = 1

    # testing begin
    img_raw = cv2.imread(args.image, cv2.IMREAD_COLOR)

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

    # save image
    if args.save_image:
        draw_keypoint(img_raw, dets, args.vis_thres)

        splits = args.image.split(".")
        name = ".".join(splits[:-1])
        ext = splits[-1]
        output = f"{name}_results.{ext}"
        cv2.imwrite(output, img_raw)


if __name__ == "__main__":
    main()
