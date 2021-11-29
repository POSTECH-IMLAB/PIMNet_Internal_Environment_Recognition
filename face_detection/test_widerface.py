#!/usr/bin/env python
import argparse
import os

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from model.prior_box import PriorBox
from model.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.misc import draw
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument(
    '--checkpoint', type=str,
    default='./weights/mobilenet0.25_final.pt',
    help='Trained state_dict file path to open'
)
parser.add_argument('--origin-size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save-folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--jit', action="store_true", default=False, help='Use JIT')
parser.add_argument('--dataset-folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence-threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top-k', default=5000, type=int, help='top_k')
parser.add_argument('--nms-threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep-top-k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save-image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis-thres', default=0.5, type=float, help='visualization_threshold')


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
    if args.jit:
        net = torch.jit.script(net)
    print('Finished loading model!')
    torch.backends.cudnn.benchmark = True

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)
    os.makedirs("./results/", exist_ok=True)

    target_size = 1600.0
    max_size = 2150.0
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = target_size / im_size_min
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
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
        _t['misc'].toc()

        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print(f"im_detect: {i+1:d}/{num_images:d}"
              f"forward_pass_time: {_t['forward_pass'].average_time:.4f}s misc: {_t['misc'].average_time:.4f}s")

        # save image
        if args.save_image:
            draw(img_raw, dets, args.vis_thres)

            # save image
            cv2.imwrite(f"./results/{i:05d}.jpg", img_raw)


if __name__ == "__main__":
    main()
