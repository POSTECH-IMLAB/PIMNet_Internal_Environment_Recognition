# Stage-wise Face Alignment using Global and Local Regressors

This is a caffe-python implementation on Windows 10 for face alignment.

We implemented two-kind of methods.<br>

Method1 repeat global and local regression after initialization regression<br>
<p align="center"><img src="figure/overview1.png" alt="" width="400"></p>

Method2 repeat local refinement regression after initialization regression<br>
<p align="center"><img src="figure/overview2.png" alt="" width="400"></p>

## Evaluation on 300w public test set
<center>

| Method | Common | Challenging | Full |
|:-------|:--------:|:-----:|:-------:|
| Stage(Projection) | 8.24 | 12.56 | 9.07 |
| Stage(Adjustment) | 6.25 | 10.16 | 7.02 |
| Stage(Global1) | 4.66 | 8.20 | 5.35 |
| Stage(Local1) | 3.45 | 6.49 | 4.05 |
| Stage(Global2) | 3.59 | 6.62 | 4.18 |
| Stage(Local2) | 3.29 | 6.14 | 3.85 |
| Stage(Global3) | 3.48 | 6.37 | 4.05 |
| Stage(Local3) | 3.28 | 6.09 | 3.83 |
| Regression(Wild, simple net) | 4.07 | 6.90 | 4.62 |
| Regression(Wild, ResNet50) | 3.72 | 6.44 | 4.25 |
</center>

## Usage

### For Training
1. Clone the repository
```
git clone https://github.com/hyunsungP/facelignmentregression
```

2. make data files (.h5)
```
make_wild_input.py
```
and so on.

3. make data file list \
Refer to models/list_train_*.txt

4. training \
On console window with caffe
```
caffe train --solver=models/ZF_solver.prototxt --gpu=0
```

Other network are same.

### For Testing
Change prototxt path in the source code.
```
test_300w_public.py
```

Other models will be uploaded.

