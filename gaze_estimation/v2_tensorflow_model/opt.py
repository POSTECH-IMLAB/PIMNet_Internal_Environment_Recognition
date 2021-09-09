import tensorflow as tf
import numpy as np
import random
import math
from glob import glob
from PIL import Image

import tensorflow.contrib.slim as slim


BATCH_SIZE = 256
IMG_WIDTH = 120
IMG_HEIGHT = 100
CHANNEL_N  = 1
CLASS_N = 6

def load_img_and_label_from_npy(image_npy, label_npy):
    images = load_np(image_npy)
    labels = load_np(label_npy)

    return images, labels

def load_images(train_ratio=0.95, test_ratio=0.05):
    print("Loading Images...")

    #응시영역 레이블별로 읽도록 한다.
    #6 gaze zones 
    data_list_1 = glob('*part_1.jpg') #1
    data_list_2 = glob('*part_3.jpg') #2
    data_list_3 = glob('*part_6.jpg')  #3
    data_list_4 = glob('*part_8.jpg')  #4
    data_list_5 = glob('*part_10.jpg')  #5
    data_list_6 = glob('*part_12.jpg')  #6


    batch_tuple = []

    n = 0
    #------------1
    for i in range(len(data_list_1)):
        path = data_list_1[i]
        img = read_image(path)

        #불러온 이미지 batch에 저장
        batch_tuple.append((path, 0))


    #-------------- 2
    for i in range(len(data_list_2)):
        path = data_list_2[i]
        img = read_image(path)

        #불러온 이미지 batch에 저장
        batch_tuple.append((path, 1))

    #--------------- 3
    for i in range(len(data_list_3)):
        path = data_list_3[i]
        img = read_image(path)

        #불러온 이미지 batch에 저장
        batch_tuple.append((path, 2))

    # 4
    for i in range(len(data_list_4)):
        path = data_list_4[i]
        img = read_image(path)

        #불러온 이미지 batch에 저장
        batch_tuple.append((path, 3))

    # 5
    for i in range(len(data_list_5)):
        path = data_list_5[i]
        img = read_image(path)

        #불러온 이미지 batch에 저장
        batch_tuple.append((path, 4))

    # 6
    for i in range(len(data_list_6)):
        path = data_list_6[i]
        img = read_image(path)

        #불러온 이미지 batch에 저장
        batch_tuple.append((path, 5))


    #섞은 후에 저장된 tuple을 풀어낸다
    random.shuffle(batch_tuple)
    #print(batch_tuple)

    #train:test 나눈다
    num = len(batch_tuple)
    train_num =  math.floor(train_ratio*num)
    test_num = num - train_num


    #트레인, 테스트 나눔
    train_batch = batch_tuple[0:train_num]
    test_batch = batch_tuple[train_num:num]
    print(len(train_batch))

    # 이미지를 numpy 형태로 받아야 한다.
    # BATCH_SIZE = len(data_list)

    train_image = np.zeros((train_num, IMG_HEIGHT, IMG_WIDTH, CHANNEL_N))
    train_label = np.zeros((train_num, CLASS_N))
    test_image = np.zeros((test_num, IMG_HEIGHT, IMG_WIDTH, CHANNEL_N))
    test_label = np.zeros((test_num, CLASS_N))

    # [TRAINING] numpy로 변환
    bat_idx = 0
    for path, label in train_batch:
        img = read_image(path)
        train_image[bat_idx,:, :,:] = img
        train_label[bat_idx, label] = 1
        bat_idx += 1

    # [TEST] numpy로 변환
    bat_idx = 0
    for path, label in test_batch:
        img = read_image(path)
        test_image[bat_idx, :, :, :] = img
        test_label[bat_idx, label] = 1
        bat_idx += 1

    print('[train_img]')
    print(train_image.shape)
    print('[test_img]')
    print(test_image.shape)
    print('[train_label]')
    print(train_label.shape)
    print('[test_label]')
    print(test_label.shape)

    save_np('train_img', train_image)
    save_np('train_label', train_label)
    save_np('test_img', test_image)
    save_np('test_label', test_label)




def save_np(filename, data):
    np.save(filename, data)

def load_np(filename):
    print('loading ' + filename + '......')
    return np.load(filename)

def read_image_and_label(path):
    return read_image(path), read_label(path)

def read_image(path):
    image = np.array(Image.open(path).convert('L'))
    image = image.astype(np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=2)
    #image = image.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
    return image


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


# conv layer
def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

# relu를 수행한다.
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)