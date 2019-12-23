#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-22 下午1:52
# @Author  : tangming
# @File    : 1.py
'''
2018年5月21日15:21:22

配置文件：
    主要解决interface 和 Hit_itraker这两个文件的配置参数一致问题
'''

class Config():
    '''
    参数配置
    '''
    # 校准参数配置
    RUN_CAL = False  # 是否要获得校准图片
    RUN_TRAIN = False  # 是否训练模型
    imagefile = './img6x6_finetune_test0521'  # 校正图片位置
    # 模型参数配置
    lr = 5e-5
    batch_size = 16
    MAX_ITER = 50  # 最大训练迭代次数
    ACCEPT_ACC = 0.80  # 训练准确率，当acc大于这个值是，认为overfit一次，overfit达到50次后会提前结束训练

    # 图像RGB均值
    cumt_picmean = [103.939, 116.779, 123.68]
    modelpath = 'model_0507/Dense22_3X3/Dense22_finetune.ckpt-6300'  # 预训练模型
    newmodelpath = 'model_0507/Dense166_final_finetune'  # 微调后的模型

    # 相机配置
    SRC_W, SRC_H = 1920, 1080  # 显示屏分辨率 （1920,1080）（1366,768）
    CAM_NUM = 0  # 脸部相机 0表示内置相机，1表示外置相机
    frontalface_cascade = 'haarcascade_frontalface_default.xml'
    eye_cascade = 'haarcascade_eye.xml'
    CAM_NUM_ENV = 0  # 环境相机暂时设置为相同的相机，方便测试

    # 网络参数设置
    # dense22
    K = 12
    L = 2
    THEATA = 0.8
    OUTPUTCLASS = 9
    KEEP_PROB = 0.5
    # 预测参数设置
    Video_file = 'output.avi'  # 将预测界面保存成一个flv文件
    Video_HW = (640, 480)
    MEAN_PICNUM = 5  # 平均这么多张图片的预测结果，再输出
    WAITPIC_NUM = 20
    # 服务器地址
    HOST = '172.27.34.1'
    PORT = 1234

    # Interface中用到的配置参数,这两个参数主要是在获取数据集的时候用
    JUST_DOWN = False  # 是否只生成屏幕下方区域的图片（补偿一些眼部无法识别的情况）
    CAM_2 = None  # None表示只用一个摄像头,1表示再加上外置摄像头（同一场景使用两个不同位置的相机获取图片）

