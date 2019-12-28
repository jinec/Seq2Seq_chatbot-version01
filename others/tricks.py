#!/usr/bin/env python3
#encoding=utf-8

##（1）关于 多GPU参数共享中的 “.name"的学习
# list1=['一','二']
# input_feed = {}
# for j in range(len(list1)):
#     input_feed[list1[j]]=list1[j]
# print(input_feed)

# #（2）placeholder搭建的架子很大，但是可以只赋予一部分值就行，只要最后run的也是一部分！
import tensorflow as tf
# decoder_weights=[]
# for i in range(30 + 1):  #位置：0：30
#     x=tf.placeholder("int32")
#     decoder_weights.append(x)
# input_feed = {}
# for i in range(10):
#     input_feed[decoder_weights[i].name] = i
# with tf.Session() as sess:
#     print(sess.run(decoder_weights[:10], input_feed))  #这里截断就可以了！
 
# #（3）tf.gradients()
# import numpy as np
# import tensorflow as tf
# 
# 
# sess = tf.Session()
# 
# x_input = tf.placeholder(tf.float32, name='x_input')
# y_input = tf.placeholder(tf.float32, name='y_input')
# w = tf.Variable(2.0, name='weight')
# b = tf.Variable(1.0, name='biases')
# y = tf.add(tf.multiply(x_input, w), b)
# loss_op = tf.reduce_sum(tf.pow(y_input - y, 2)) / (2 * 32)
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)
# 
# '''tensorboard'''
# gradients_node = tf.gradients(loss_op, w)
# print(gradients_node)
# tf.summary.scalar('norm_grads', gradients_node)
# tf.summary.histogram('norm_grads', gradients_node)
# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('log')
# 
# init = tf.global_variables_initializer()
# sess.run(init)
# 
# '''构造数据集'''
# x_pure = np.random.randint(-10, 100, 32)
# x_train = x_pure + np.random.randn(32) / 10  # 为x加噪声
# y_train = 3 * x_pure + 2 + np.random.randn(32) / 10  # 为y加噪声
# 
# for i in range(20):
#     _, gradients, loss = sess.run([train_op, gradients_node, loss_op],
#                                   feed_dict={x_input: x_train[i], y_input: y_train[i]})
#     print("epoch: {} \t loss: {} \t gradients: {}".format(i, loss, gradients))
# 
# sess.close()

#(4)tf.clip_by_norm

with tf.Graph().as_default():
    x = tf.Variable(initial_value=3., dtype='float32')
    w = tf.Variable(initial_value=4., dtype='float32')
    y = w*x
    
    opt = tf.train.GradientDescentOptimizer(0.1)
    grads_vals = opt.compute_gradients(y, [w])
    for i, (g, v) in enumerate(grads_vals):
        if g is not None:
            grads_vals[i] = (tf.clip_by_norm(g, 0.5), v)  # clip gradients
            #设置的值大了，就用自己的；小了，就乘以比率降下来！
    train_op = opt.apply_gradients(grads_vals)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(grads_vals))
        print(sess.run([x,w,y]))