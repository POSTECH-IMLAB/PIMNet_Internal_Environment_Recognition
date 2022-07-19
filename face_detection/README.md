# Face Detection (work in progress)
The code and checkpoints contained in this repository were adopted from the [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) repository.


## Getting Started

### Requirements / Installation
- [Anaconda](https://www.anaconda.com/)
- Nvidia GPU (for GPU utilization)

Use the following commands to install the necessary packages and activate the environment:
```sh
conda env create -f environment.yml
conda activate retinaface
```

### Data
1. Download the [WiderFace](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA).

3. Organise the dataset directory as follows:

```
./data/widerface/
├─train/
│  ├─images/
│  └─label.txt
└─val/
   ├─images/
   └─wider_val.txt
```

ps: wider_val.txt only include val file names but not label information.


### Test
You can use the following command to detect faces in a photo and save the result as an image:
```sh
python detect.py --image <path to image file> -s
```
See [detect.py](detect.py#L16) for available arguments.


## Training
We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58%  in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as follows:
```bash
./weights/
├─mobilenet0.25_final.pt
└─mobilenet0.25_pretrain.tar
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WiderFace:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobilenet0.25
  ```


## Evaluation

### Evaluation WiderFace val
1. Generate txt file
```Shell
python test_widerface.py --trained-model <weight file> --network mobilenet0.25 or resnet50
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use WiderFace official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)

### Evaluation FDDB

1. Download the images [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
./data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
python test_fddb.py --trained_model <weight file> --network mobilenet0.25 or resnet50
```

3. ~~Download [eval_tool](https://bitbucket.org/marcopede/face-eval) to evaluate the performance.~~ This link doesn't seem to work anymore. We found [this](https://github.com/RuisongZhou/FDDB_Evaluation) repository, but haven't tested it yet.


## References and Citation
- [RetinaFace in PyTorch](https://github.com/biubug6/Pytorch_Retinaface)
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

```
@inproceedings{deng2020retinaface,
  title={RetinaFace: Single-Shot Multi-Level Face Localisation in the wild},
  author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5203--5212},
  year={2020}
}
