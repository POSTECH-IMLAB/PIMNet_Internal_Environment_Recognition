#import tensorflow as tf
from opt import  *
from model import gazenetwork
import random
import math
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import time



BATCH_SIZE = 256
IMG_WIDTH = 120
IMG_HEIGHT = 100
CHANNEL_N  = 1
CLASS_N = 6


def predict_imgs():
    tf.logging.set_verbosity(tf.logging.INFO)
    # to avoid cuda memory out error
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # data load
    face_npy, img_list = load_imgs()
    IMG_NUM = len(img_list)

    # estimator 선언
    gaze_classifier = tf.estimator.Estimator(model_fn=gazenetwork, model_dir="./model",
                                             config=tf.contrib.learn.RunConfig(session_config=config))



    # START
    img_template = None
    for  i in range(IMG_NUM):
        test_data = face_npy[i, :, :, :]
        test_data = np.expand_dims(test_data, axis=0)

        # test
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            shuffle=False)
        #test_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)

        predictions = gaze_classifier.predict(input_fn=test_input_fn)
        predictor = list(predictions)
        label = predictor[0]['classes'] + 1

        #draw pic
        draw_pic(img_template, img_list[i], label, i)

        #print(list(predictions)[0]['claasses'])




def draw_pic(img_template, img_path, text, frameidx):


    plt.gcf().clear()
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    (x, y) = (10, 10)
    font = ImageFont.truetype('arial', size=125)
    message = str(text)
    color = 'rgb(255, 255, 255)'  # black color
    draw.text((x, y), message, fill=color, font=font)
    #plt.imshow(image)

    if img_template is None:
        img_template = plt.imshow(image)
    else:
        img_template.set_data(image)

    plt.pause(0.1)

    #im = plt.imshow(image, animated=True)
    plt.draw()



'''
def load_imgs():
    BASE_DIR = "F:/2-2/cv/proj_gaze/sequences/4"
    face_dir = BASE_DIR + "/face/*.jpg"
    img_dir = BASE_DIR + "/entire/*.jpg"

    face_list = glob(face_dir)
    img_list = glob(img_dir)

    IMG_NUM = len(img_list)
    test_image = np.zeros((IMG_NUM, IMG_HEIGHT, IMG_WIDTH, CHANNEL_N))

    # LOOP START
    bat_idx = 0
    for path in face_list:
        img = read_image(path)
        test_image[bat_idx,:, :,:] = img
        bat_idx += 1




    return test_image, img_list
'''

def read_image(path):
    image = np.array(Image.open(path).convert('L'))
    image = image.astype(np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=2)
    return image

# main func
def main(unused_argv):
    #load_imgs()
    predict_imgs()


if __name__ == "__main__":
    tf.app.run()