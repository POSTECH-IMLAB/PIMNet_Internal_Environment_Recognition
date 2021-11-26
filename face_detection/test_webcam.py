#!/usr/bin/env python
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision.ops import nms

from data import cfg_mnet, cfg_re50
from model.prior_box import PriorBox
from model.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from utils.utils import load_model

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('--checkpoint', default='./weights/mobilenet0.25_final.pt',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobilenet0.25', choices={"mobilenet0.25", "resnet50"})
parser.add_argument('--save-folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence-threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top-k', default=5000, type=int, help='top_k')
parser.add_argument('--nms-threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep-top-k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save-image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis-thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def main():
    if args.network == "mobilenet0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    # net and model
    net = RetinaFace(**cfg)
    net = load_model(net, args.checkpoint, args.cpu, is_train=False)
    # net = torch.jit.script(net)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = 'data/FDDB/images/'
    testset_list = 'data/FDDB/img_list.txt'
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing scale
    resize = 1

    _t = {
        "preprocess": Timer(),
        "inference": Timer(),
        "nms": Timer(),
        "misc": Timer(),
    }

    # testing begin
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    while True:
        ret, img_raw = cap.read()

        # NOTE preprocessing.
        _t["runtime"].tic()
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        # NOTE inference.
        loc, conf, landms = net(img)  # forward pass

        # NOTE misc.
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
        _t["runtime"].tic()

        print(f"runtime: {_t['preprocess'].average_time:.4f} sec/iter")

        # show image
        for b in dets[:5]:
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

        cv2.imshow("Face Detection Demo", img_raw)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

