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
parser.add_argument('--save-folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--jit', action="store_true", default=False, help='Use JIT')
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
        "forward": Timer(),
        "postprocess": Timer(),
        "misc": Timer(),
    }

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name + '.jpg'
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # NOTE preprocessing.
        _t["preprocess"].tic()
        img = img_raw - (104, 117, 123)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.as_tensor(
            [im_width, im_height, im_width, im_height],
            dtype=torch.float, device=device
        )
        img = img.transpose(2, 0, 1)
        img = np.float32(img)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        _t["preprocess"].toc()

        # NOTE forward.
        _t["forward"].tic()
        loc, conf, landms = net(img)  # forward pass
        _t["forward"].toc()

        # NOTE misc.
        _t["postprocess"].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        scores = conf.squeeze(0)[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.as_tensor(
            [im_width, im_height] * 5, dtype=torch.float, device=device
        )
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
        _t["postprocess"].toc()

        # do NMS
        _t["misc"].tic()
        keep = nms(boxes, scores, args.nms_threshold)
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
        fw.write(f'{img_name:s}\n')
        fw.write(f'{dets.shape[0]:.1f}\n')
        for k in range(dets.shape[0]):
            xmin, ymin, xmax, ymax = dets[k, :4]
            score = dets[k, 4]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            # fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
            fw.write('{:d} {:d} {:d} {:d} {:.10f}\n'.format(int(xmin), int(ymin), int(w), int(h), score))

        print(
            f"im_detect: {i+1:d}/{num_images:d}\t"
            f"preprocess_time: {_t['preprocess'].average_time:.4f}s\t"
            f"forward_time: {_t['forward'].average_time:.4f}s\t"
            f"postprocess_time: {_t['postprocess'].average_time:.4f}s\t"
            f"misc_time: {_t['misc'].average_time:.4f}s"
        )

        # show image
        if args.save_image:
            draw(img_raw, dets, args.vis_thres)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            cv2.imwrite(f"./results/{i:05d}.jpg", img_raw)

    fw.close()


if __name__ == "__main__":
    main()
