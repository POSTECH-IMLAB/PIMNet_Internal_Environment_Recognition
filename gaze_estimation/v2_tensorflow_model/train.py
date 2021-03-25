import tensorflow as tf
from opt import  *
from model import gazenetwork

def test():
    tf.logging.set_verbosity(tf.logging.INFO)
    # to avoid cuda memory out error
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # data load
    eval_data, eval_label = load_img_and_label_from_npy('test_img.npy', 'test_label.npy')
    eval_label = np.argmax(eval_label, axis=1)
    print(eval_data[3])
    print(eval_label[3])
    print('npy loaded')

    # estimator 선언
    gaze_classifier = tf.estimator.Estimator(model_fn=gazenetwork, model_dir="./model",
                                             config=tf.contrib.learn.RunConfig(session_config=config))

    # eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_label,

        num_epochs=1,
        shuffle=False)
    eval_results = gaze_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

def train():
    # load_images()

    tf.logging.set_verbosity(tf.logging.INFO)
    # to avoid cuda memory out error
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # == ESTIMATOR 에 들어갈 input_fn 역시 조건 있다.
    # input_fn의 조건은 datrue과 label data 반환을 목적으로 한다
    train_data, train_label = load_img_and_label_from_npy('train_img.npy', 'train_label.npy')
    train_label = np.argmax(train_label, axis=1)
    print('npy loaded')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_label,
        batch_size=712, num_epochs=None, shuffle=True)
    print('input_fn craeated')

    # == ESTIMATOR 학습을 위한 model_fn에 파라미터 등등에 대해 조건이 필요하다
    # <arg>
    # (features, labels, mode, params, config) 인데, features와 labels는 반드시 필수
    # <return>
    # tf.estimator.EstimatorSpecwor

    # == ESTIMATOR의 model_dir은 학습 파라미터가 저장된다 그리고 config도 들어가고..
    gaze_classifier = tf.estimator.Estimator(model_fn=gazenetwork, model_dir="./model",
                                             config=tf.contrib.learn.RunConfig(session_config=config))
    print('estimator craeated')

    # recording logs
    log_tensor = {"loss" : "loss"}
    #logging_hook = tf.train.LoggingTensorHook({"loss": loss,
    #                                           "accuracy": accuracy}, every_n_iter=10)

    log_hook = tf.train.LoggingTensorHook(tensors=log_tensor, every_n_iter=50)

    # train
    print('start train')
    gaze_classifier.train(input_fn=train_input_fn, steps=100000)

def make_db():
    load_images()

#main func
def main(unused_argv):
    #make_db()
    #test()
    train()



if __name__ == "__main__":
    tf.app.run()