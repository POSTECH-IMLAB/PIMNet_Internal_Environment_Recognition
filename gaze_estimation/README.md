# IR_Driver_Gaze_Estimation

Implementation of gaze estimation using IR camera images with CNN. 

In this repository, light model version of gaze estimation (caffe, tensorflow and pytorch) and heavy model version

* input : 120 x 100 grayscale face image
* Light version : use 120 x 100 grayscale image for global estimator
* Heavy version : use 120 x 100 grayscale image for global estimator and crop it to 80 x 100 image for local estimator
* Heavy+Att version : add attention mask to heavy version


## CAFFE version
Light model version is supported

-TRAINING from Scratch-
> bin\caffe train --solver=ir_gaze_solver.prototxt --gpu=0

-TRAINING from Weights-
> bin\caffe train --solver=ir_gaze_solver.prototxt --weights=caffemodels/***.caffemodel --gpu=0



## TENSORFLOW version
Light model version is supported

-TRAINING/EVALUATION from Scratch-
> python train.py 

-PREDICT-
>python test_sequences.py



## PYTORCH version
Modify config.py for various options (such as batch size, gpu index, ..)

-TRAINING-
> python train.py