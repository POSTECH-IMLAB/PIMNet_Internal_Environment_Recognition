import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from opt import *



def gazenetwork(features, labels, mode):

    # image : [batch, 100, 120, 1]
    input = tf.reshape(features["x"], [-1, 100, 120, 1])

    # 트레이닝 때는 드롭아웃 적용
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout = 0.5
    else:
        dropout = 1.0



    # * conv는 기본적으로 SAME PADDING
    # H0
    h0 = lrelu(conv2d(input, output_dim=40, ks=7, s=2, name='h0_conv'))
    h0 = slim.max_pool2d(h0, kernel_size=3, stride=2, scope='h0_pool')

    # H1
    h1 = lrelu(conv2d(h0, output_dim=70, ks=5, s=2, name='h1_conv'))
    h1 = slim.max_pool2d(h1, kernel_size=2, stride=2, scope='h1_pool')

    # H2
    h2 = lrelu(conv2d(h1, output_dim=60, ks=3, s=1, name='h2_conv'))
    h2 = slim.max_pool2d(h2, kernel_size=2, stride=2, scope='h2_pool')

    # H3
    h3 = lrelu(conv2d(h2, output_dim=80, ks=3, s=1, name='h3_conv'))
    h3 = slim.max_pool2d(h3 , kernel_size=2, stride=2, scope='h3_pool')

    # H4
    h4 = lrelu(conv2d(h3, output_dim=100, ks=3, s=1, name='h4_conv'))

    # h3 & h4 concatenate
    h3_flat = slim.flatten(h3, scope="h3_flat")
    h4_flat = slim.flatten(h4, scope="h4_flat")
    h_concat = tf.concat([h3_flat, h4_flat], 1, name='h3_h4_concat')

    # start of fc
    fc1 = slim.fully_connected(h_concat, 4000, scope="fc1")
    fc1_dropout = slim.dropout(fc1, dropout)
    logits = slim.fully_connected(fc1_dropout, 6, activation_fn=None, scope="logits")
    class_logits = tf.argmax(input=logits, axis=1)

    # softmax 거침
    #softmax
    predictions = {"classes" : tf.argmax(input=logits, axis=1),
                "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")}
    #predictions = tf.nn.softmax(logits, name='predictions')




    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)


    #in TRAINING mode,
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        # Training 모드의 EstimatorSpec을 출력해야 한다. EstimatorSpec은 mode, loss, train_op를 포함하여야 한다.
        # train_po는 loss의 optimizer을 minimization 하는 것
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #in PREDICT mode.
    if mode == tf.estimator.ModeKeys.PREDICT:
        out_predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        #out_predictions = {"logits": logits}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=out_predictions)

    #in EVAL mode.
    print(labels)
    print(class_logits)
    eval_ops = {"accuracy" : tf.metrics.accuracy(labels=labels, predictions=class_logits)}

    # Eval 모드의 EstimatorSpec을 출력해야 한다. EstimatorSpec은 mode, loss, eval_ops를 포함하여야 한다.
    # eval은 accuracy
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_ops)








