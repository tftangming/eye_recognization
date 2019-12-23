#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-22 下午1:52
# @Author  : tangming
'''
    说明：
        2017年11月21日15:35:26
        cnn的常用模块，建立新的模型的时候可以继承这个类，省去许多基础的函数定义
            --   _convlayer()               :实现卷积
            --  _poollayer()                :实现池化
            --  _convblock()                :bn+relu+conv的组合操作
            --  save_network_weight()       :提取网络中所有的可训练函数，保存为pkl文件
            --  init_network()              :利用pkl文件初始化网络中所有的可训练参数
            --  get_l2loss()                :计算所有可训练参数的l2 norm,累加后返回这个值
            --  count_trainable_params()    :统计所有可训练参数数量占用的存储空间
'''
import tensorflow as tf
import numpy as np


class bisic_cnn(object):
    def __init__(self, sess):
        print('cnn basic module inherit!')

    def _convlayer(self,x,name,shape,padding='VALID',stride=[1,1]):
        '''
        conv函数
        If padding == "SAME":
        output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
        If padding == "VALID":
        output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) ) / strides[i])
        :param x:输入
        :param name: 模块名称
        :param shape: [filter_h,filter_w,input_channel,output_channel,]
        :param stride:卷积的h,w的步长
        :param padding:
        :return:
        '''
        with tf.variable_scope(name):
            w = tf.get_variable('weight', shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable= False)
            b = tf.get_variable('biases', shape=shape[-1:], initializer=tf.constant_initializer(0.01))
            conv_ = tf.nn.conv2d(x, w, [1, stride[0], stride[1], 1], padding=padding, name='conv')
            conv_ = tf.nn.bias_add(conv_, b, name='bias_add')
            return conv_

    def _poollayer(self, input_x, pooling='max', size=(3, 3), stride=(2, 2), padding='SAME'):
        '''
        pool函数
        :param input:输入
        :param pooling: 'avg'表示均值池化，用于最后输出score.'max'表示maxpool
        :param size:
        :param stride:
        :param padding:
        :return:
        '''
        if pooling == 'avg':
            x = tf.nn.avg_pool(input_x, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                               padding=padding)
        else:
            x = tf.nn.max_pool(input_x, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                               padding=padding)
        return x

    def _convblock(self, x, name, shape, bn_istraing, padding='VALID', stride=[1, 1], mode='conv_bn_relu'):
        '''
        标准的：CONV->BN -> RELU
        三个函数结合
        :param x:输入
        :param name: 模块名称
        :param shape: [filter_h,filter_w,input_channel,output_channel,]
        :param bn_istraing:是否是在训练
        :param stride:卷积的h,w的步长
        :param padding:
        :param mode: bn ->relu->conv 顺序（'bn_relu_conv'）或者是 conv-> bn-> relu 顺序（'conv_bn_relu'）
        :return:
        '''
        block_output=None
        with tf.variable_scope(name):
            # param
            with tf.name_scope('weight_biases'):
                w = tf.get_variable('weight', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('biases', shape=shape[-1:], initializer=tf.constant_initializer(0.01))
            if mode == 'conv_bn_relu':
                # conv
                conv_ = tf.nn.conv2d(x, w, [1, stride[0], stride[1], 1], padding=padding, name='conv')
                conv_ = tf.nn.bias_add(conv_, b, name='bias_add')
                # bn
                bn_ = tf.layers.batch_normalization(conv_, training=bn_istraing, name='BN')
                # relu
                block_output = tf.nn.relu(bn_, name='relu')
            elif mode == 'bn_relu_conv':
                # BN
                bn_ = tf.layers.batch_normalization(x, training=bn_istraing, name='BN')
                # relu
                relu_ = tf.nn.relu(bn_, name='relu')
                # conv
                conv_ = tf.nn.conv2d(relu_, w, [1, stride[0], stride[1], 1], padding=padding, name='conv')
                block_output = tf.nn.bias_add(conv_, b, name='bias_add')
            else:
                raise ValueError('mode can only be conv_bn_relu or bn_relu_conv !!!')
        return block_output

    def save_network_weight(self, filename, sess):
        '''
        提取网络的权值，并保存到文件
        :param filename: 文件名
        :return:
        '''
        import pickle
        import os
        file_dir = os.path.split(filename)[0]
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
            print('Create file:{}'.format(file_dir))
        print('extracting network weight to file:{}'.format(filename))
        all_var_name = list(tf.trainable_variables())
        weight_dict = {}
        # 提取变量值
        for name_ in all_var_name:
            # 变量名称
            layer_name = str(name_).split("'")[1][:-2]
            print(layer_name)
            with tf.variable_scope('', reuse=True):
                var=tf.get_variable(layer_name)
                # 注意var是tensor，需要转换一下
                weight_dict[layer_name] = sess.run(var)
        # 保存到pkl文件中
        fp = open(filename,'wb')
        pickle.dump(obj=weight_dict, file=fp)
        fp.close()
        print('save weight file done!')

    def init_network(self, weight_addr, sess, skip_layer=[]):
        '''
        利用保存好的权值文件来初始化网络
        ！！注意假如需要调用这个函数，一定不可以在调用这个函数后在对全部变量进行初始化 （sess.run(init)）！！
        这样会使得加载的权值被覆盖，要先进行全局变量初始化再读取权值文件！
        ！！当网络中有BN的时候，其一些参数不会被保存，所以除非是迁移学习，否则不要使用这个函数来保存模型
        :param weight_addr:权值文件
        :return:
        '''
        # 全局变量初始化
        init = tf.global_variables_initializer()
        sess.run(init)
        # 加载权值文件,写入网络
        print('loading weight file:{}'.format(weight_addr))
        network_dict = np.load(weight_addr)
        layer_name = list(network_dict.keys())
        for name_ in layer_name:
            if name_ in skip_layer:
                print('skip layer:{}'.format(name_))
                continue
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(name_)
                sess.run(var.assign(network_dict[name_]))
        print('network init done!')

    def get_l2loss(self):
        '''
        计算L2 LOSS,取所有可训练参数
        :return:
        '''
        with tf.name_scope('Reg_Loss'):
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        return l2_loss

    def count_trainable_params(self):
        '''
        统计所有可训练参数
        :return:
        '''
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        # float浮点型数据类型，占4个字节
        print("Total training params: %.5f Million,%.5f Mb" % (total_parameters / 1e6, total_parameters * 4 / (1024*1024)))