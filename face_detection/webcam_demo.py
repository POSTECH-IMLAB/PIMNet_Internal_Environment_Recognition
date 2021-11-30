#!/usr/bin/env python
import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from model.prior_box import PriorBox
from model.retinaface import RetinaFace
from utils.misc import draw, inference
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument(
    '--checkpoint', type=str,
    default='./weights/mobilenet0.25_final.pt',
    help='Trained state_dict file path to open'
)
parser.add_argument(
    '--cpu', action="store_true", default=False,
    help='Use cpu inference'
)
parser.add_argument(
    '--jit', action="store_true", default=False,
    help='Use JIT'
)
parser.add_argument(
    '--confidence-threshold', type=float, default=0.02,
    help='confidence_threshold'
)
parser.add_argument(
    '--nms-threshold', type=float, default=0.4,
    help='nms_threshold'
)
parser.add_argument(
    '--vis-thres', type=float, default=0.5,
    help='visualization_threshold'
)
parser.add_argument(
    '-s', '--save-image', action="store_true", default=False,
    help='show detection results'
)
parser.add_argument(
    '--save-dir', type=str, default='demo',
    help='Dir to save results'
)


def main():
    args = parser.parse_args()
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
        dets = inference(
            net, img_raw, scale, scale1, prior_data, cfg,
            args.confidence_threshold, args.nms_threshold, device
        )
        timer.toc()

        print(f"runtime: {timer.average_time:.4f} sec/iter")

        # show image
        draw(img_raw, dets, args.vis_thres, timer.diff)

        if args.save_image:
            cv2.imwrite(fname.format(nframe), img_raw)
            nframe += 1

        cv2.imshow("Face Detection Demo", img_raw)
        if cv2.waitKey(1) == 27:  # Press ESC button to quit.
            break

        ret_val, img_raw = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()