#!/usr/bin/env python
import argparse
import os

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
parser.add_argument('-m', '--trained-model', default='./weights/mobilenet0.25_final.pt',
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
    net = load_model(net, args.trained_model, args.cpu, is_train=False)
    # net = torch.jit.script(net)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    # save file
    os.makedirs(args.save_folder, exist_ok=True)
    fw = open(os.path.join(args.save_folder, 'FDDB_dets.txt'), 'w')


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
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name + '.jpg'
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # NOTE preprocessing.
        _t["preprocess"].tic()
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
        _t["preprocess"].toc()

        # NOTE inference.
        _t["inference"].tic()
        loc, conf, landms = net(img)  # forward pass
        _t["inference"].toc()

        # NOTE misc.
        _t["misc"].tic()
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
        _t["nms"].tic()
        keep = nms(boxes, scores, args.nms_threshold)
        _t["nms"].toc()

        boxes = boxes[keep]
        scores = scores[keep]
        landms = landms[keep]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        landms = landms.cpu().numpy()
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        dets = np.concatenate((dets, landms), axis=1)
        _t["misc"].toc()

        # save dets
        fw.write('{:s}\n'.format(img_name))
        fw.write('{:.1f}\n'.format(dets.shape[0]))
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            # fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
            fw.write('{:d} {:d} {:d} {:d} {:.10f}\n'.format(int(xmin), int(ymin), int(w), int(h), score))

        print(
            f"im_detect: {i+1:d}/{num_images:d}\t"
            f"preprocess_time: {_t['preprocess'].average_time:.4f}s\t"
            f"inference_time: {_t['inference'].average_time:.4f}s\t"
            f"nms_time: {_t['nms'].average_time:.4f}s\t"
            f"misc_time: {_t['misc'].average_time:.4f}s"
        )

        # show image
        if args.save_image:
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
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            cv2.imwrite(f"./results/{i:05d}.jpg", img_raw)

    fw.close()


if __name__ == "__main__":
    main()
