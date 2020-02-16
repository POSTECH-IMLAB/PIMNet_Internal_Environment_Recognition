## Requirement

* cuda 10.2
* Python 3.7
* Pytorch 1.1
* matplotlib
* skimage
* opencv for python3

## Train

Training dataset folder (./data/train_img)

parsing_maps(face priror) 사용 (./data/train_img/Parsing_Maps)

데이터셋 추가 후 data 폴더의 train_fileList.txt에 이미지 이름 추가가 필요합니다

Train은 파이썬 환경에서 train_main.py를 불러오는 것으로 간단하게 이루어집니다

>> python train_main.py 

트레이닝 시 command line argument를 사용하여 세팅이 가능합니다 (코드 내에서 default 값을 수정해도 무방합니다)

## Test

학습된 weight를 pretrained로 불러오고 네트워크를 세팅한 뒤 dataloader 혹은 opencv로 이미지를 불러와서 적용하면 됩니다

#해당 코드 참조
>> upscaled,boundaries,reconstructed = fsrnet(input)

commit test3
