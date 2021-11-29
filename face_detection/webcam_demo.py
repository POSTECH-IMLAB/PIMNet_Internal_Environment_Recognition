#!/usr/bin/env python
import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision.ops import nms

from model.prior_box import PriorBox
from model.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--checkpoint', default='./weights/mobilenet0.25_final.pt',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--jit', action="store_true", default=False, help='Use JIT')
parser.add_argument('--confidence-threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top-k', default=5000, type=int, help='top_k')
parser.add_argument('--keep-top-k', default=750, type=int, help='keep_top_k')
parser.add_argument('--nms-threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--vis-thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('-s', '--save-image', action="store_true", default=False, help='show detection results')
parser.add_argument('--save-dir', default='demo', type=str, help='Dir to save results')
args = parser.parse_args()


def main():
    assert os.path.isfile(args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]
    device = torch.device("cpu" if args.cpu else "cuda")

    # net and model
    net = RetinaFace(**cfg)
    net.load_state_dict(checkpoint["net_state_dict"])
    net.eval().requires_grad_(False)
    net.to(device)
    print('Finished loading model!')
    cudnn.benchmark = True

    # prepare testing
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    ret_val, img_tmp = cap.read()
    im_height, im_width, _ = img_tmp.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    scale = scale.to(device)

    scale1 = torch.Tensor([im_width, im_height] * 5)
    scale1 = scale1.to(device)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    if args.jit:
        img_tmp = img_tmp.transpose(2, 0, 1)
        img_tmp = np.float32(img_tmp)
        img_tmp = torch.from_numpy(img_tmp).unsqueeze(0)
        dummy = img_tmp.to(device)
        net = torch.jit.trace(net, example_inputs=dummy)

    timer = Timer()
    if args.save_image:
        nframe = 0
        fname = os.path.join(args.save_dir, "{:06d}.jpg")
        os.makedirs(args.save_dir, exist_ok=True)

    # testing begin
    ret_val, img_raw = cap.read()
    while ret_val:
        # NOTE preprocessing.
        timer.tic()
        img = img_raw - (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.float32(img)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)

        # NOTE inference.
        loc, conf, landms = net(img)  # forward pass

        # NOTE misc.
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes *= scale
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        landms *= scale1

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
        timer.toc()

        print(f"runtime: {timer.average_time:.4f} sec/iter")

        # show image
        for b in dets[:5]:
            if b[4] < args.vis_thres:
                continue
            text = f"{b[4]:.4f}"
            b = list(map(round, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(
                img_raw, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
            )

            text = f"{1.0 / timer.diff:.1f} fps"
            cv2.putText(
                img_raw, text, (5, 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
            )

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

        if args.save_image:
            cv2.imwrite(fname.format(nframe), img_raw)
            nframe += 1
    
        cv2.imshow("Face Detection Demo", img_raw)
        c = cv2.waitKey(1)  # Press ESC button to quit.
        if c == 27:
            break

        ret_val, img_raw = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()