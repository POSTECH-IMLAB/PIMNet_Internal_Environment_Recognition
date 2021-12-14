import cv2
import numpy as np
import torch
from torchvision.ops import nms

from .box_utils import decode, decode_landm


def draw_keypoint(image, dets, threshold):
    for b in dets:
        if b[4] < threshold:
            continue
        text = f"{b[4]:.4f}"
        b = list(map(round, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(
            image, text, (cx, cy),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
        )

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)


def inference(
    network, image, scale, scale1, prior_data,
    cfg, confidence_threshold, nms_threshold, device
):
    img = image - (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = np.float32(img)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    loc, conf, landms = network(img)  # forward pass

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes *= scale
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    landms *= scale1

    # ignore low scores
    inds = torch.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # do NMS
    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    landms = landms[keep]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    landms = landms.cpu().numpy()
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    dets = np.concatenate((dets, landms), axis=1)
    return dets