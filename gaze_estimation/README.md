# IR_Driver_Gaze_Estimation

In this repository, light version of gaze estimation is uploaded
(caffe, tensorflow and pytorch)

input : 120 x 100 grayscale face image


<CAFFE>
Based on IR camera, implmented with Caffe

-TRAINING from Scratch-
bin\caffe train --solver=ir_gaze_solver.prototxt --gpu=0

-TRAINING from Weights-
bin\caffe train --solver=ir_gaze_solver.prototxt --weights=caffemodels/***.caffemodel --gpu=0

For the test, the ir_gaze_solver.deploy can be utilized.




<TF>
-TRAINING/EVALUATION from Scratch-
python train.py 

-PREDICT-
python test_sequences.py



<PyTorch>
-TRAINING-
python train.py