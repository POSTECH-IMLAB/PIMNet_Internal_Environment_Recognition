#!/usr/bin/env python
import argparse
import os

import torch
from model.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Convert to ONNX')
parser.add_argument(
    '--checkpoint', type=str,
    default='./weights/mobilenet0.25_final.pt',
    help='Trained state_dict file path to open'
)
parser.add_argument(
    '--long-side', type=int, default=640,
    help='when origin_size is false, long_side is scaled size(320 or 640 for long side)'
)
parser.add_argument(
    '--cpu', action="store_true",
    help='Use cpu inference'
)


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

    # ------------------------ export -----------------------------
    output_onnx = 'face_detector.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, args.long_side, args.long_side).to(device)

    torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names)


if __name__ == "__main__":
    main()
