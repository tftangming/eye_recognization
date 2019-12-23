#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-22 下午1:52
# @Author  : tangming
'''
2018年5月8日09:42:50
说明：加载训练好的模型，进行前向的推断,主要特点
    1.可以对新环境进行重新的标定，也就是在新环境下得到小批量训练图片，fine-tune预训练模型
    2.
'''

import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cumt_variant_fc import densenet
import interface
import os
import time
import warnings
from SignalCorps import signalcorps,signalcorps_localtest
from Config import Config
warnings.filterwarnings('ignore')

class Hit_traker():
    '''
    Dense166类，主要功能为建立模型，加载模型，微调模型
    EXAMPLE：
        model_dense166=Hi t_traker()
        model_dense166.buildAll()
        model_dense166.finetuneNetwork(data,label)
    '''
    def __init__(self):
        self.sess=tf.Session()
        self.init=tf.global_variables_initializer()
        self.output_class=Config.OUTPUTCLASS
        print('#'*50)
        print("Building Dense166 model!")
        self._adddPlaceholder()
        self._buildNetwork()
        self._addTrainer()
        print('Loading Pre-train model!')
        self.initmodel()


    def _adddPlaceholder(self):
        '''
        加载占位符
        :return:
        '''
        with tf.variable_scope('Placehloder'):
            self.X=tf.placeholder(dtype=tf.float32,shape=[None,128,128,3],name='X')
            self.Y=tf.placeholder(dtype=tf.float32,shape=[None,self.output_class],name='Y')
            self.bn_train=tf.placeholder(dtype=tf.bool,name='BN_FLAG')
            self.LR=tf.placeholder(dtype=tf.float32,name='lr')
            self.KEEP_PROB=tf.placeholder(dtype=tf.float32,name='kp')
    def _buildNetwork(self):
        '''
        建立Dense166网络
        :return:
        '''
        with tf.name_scope('DenseNet'):
            self.model=densenet(Images=self.X,bn_istraining=self.bn_train,K=Config.K,L=Config.L,theta=Config.THEATA,
                                                                    output_class=self.output_class,sess=self.sess,denseblock_num=4,KEEP_PROB=self.KEEP_PROB)
            self.y_score=self.model.y_score
            self.prob=tf.nn.softmax(self.y_score)
    def _addTrainer(self):
        '''
        训练相关Ops
        :return:
        '''
        with tf.name_scope('LOSS'):
            self.LOSS=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_score,labels=self.Y)
                                     )
        with tf.name_scope('TRAIN'):
            self.TRAIN=tf.train.AdamOptimizer(self.LR).minimize(self.LOSS)
        with tf.name_scope('ACCURACY'):
            acc_count=tf.equal(tf.argmax(self.y_score,1),tf.argmax(self.Y,1))
            self.ACCURACY=tf.reduce_mean(tf.cast(acc_count,tf.float32))

        self.bn_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    def get_l2loss(self):
        '''
        计算L2 LOSS,取所有可训练参数
        :return:
        '''
        with tf.name_scope('Reg_Loss'):
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        return l2_loss

    def initmodel(self):
        '''
        加载训练好的模型
        :return:
        '''
        model_path=Config.modelpath
        saver=tf.train.Saver()
        saver.restore(self.sess,model_path)

    def finetuneNetwork(self,data,label):
        '''
        根据给定的数据微调网络
        :param data: 矩阵形式的眼部图像 [N,H,W,3] 没有减均值
        :param label: [N,36] one-hot标签
        :return:
        '''
        #按 8:2 划分数据集
        data=np.float32(data-Config.cumt_picmean)
        index_=np.arange(data.shape[0])
        np.random.shuffle(index_)
        tr_index=index_[:int(data.shape[0]*0.8)]
        te_index=index_[int(data.shape[0]*0.8):]

        tr_data,tr_label=data[tr_index],label[tr_index]
        te_data,te_label=data[te_index],label[te_index]

        print("train data set:{},test data set:{}".format(tr_data.shape[0],te_data.shape[0]))
        batchsize=Config.batch_size
        lr_=Config.lr
        epoch_iters=Config.MAX_ITER
        X,Y=self.X,self.Y
        KEEP_PROB=self.KEEP_PROB
        bn_train,LR,TRAIN,bn_ops=self.bn_train,self.LR,self.TRAIN,self.bn_ops

        sess,LOSS,ACCURACY=self.sess,self.LOSS,self.ACCURACY
        overfit_count=0
        for i in range(epoch_iters+1):
            mask=np.random.choice(tr_data.shape[0],batchsize,replace=False)
            x_,y_=tr_data[mask],tr_label[mask]
            feed_dict={X:x_,Y:y_,bn_train:True,LR:lr_,KEEP_PROB:Config.KEEP_PROB}
            sess.run([TRAIN,bn_ops],feed_dict=feed_dict)
            if i%10==0:
                mask=np.random.choice(tr_data.shape[0],32,replace=False)#
                x_,y_=tr_data[mask],tr_label[mask]
                feed_dict={X:x_,Y:y_,bn_train:False,KEEP_PROB:1.0}
                loss_,acc_=sess.run([LOSS,ACCURACY],feed_dict=feed_dict)
                #writer_tr.add_summary(m_,i)
                print('epoch:{},train loss:{},train accuracy:{}'.format(i,loss_,acc_))
                if acc_>Config.ACCEPT_ACC:
                    overfit_count+=1
            if i%20==0:
                mask=np.random.choice(te_data.shape[0],32,replace=False)
                x_,y_=te_data[mask],te_label[mask]
                feed_dict={X:x_,Y:y_,bn_train:False,KEEP_PROB:1.0}
                loss_,acc_=sess.run([LOSS,ACCURACY],feed_dict=feed_dict)
                print('--epoch:{},test loss:{},test accuracy:{}'.format(i,loss_,acc_))
            if overfit_count>20:
                print('#'*50)
                print("Overfit,early stop!! {}/{}".format(i,epoch_iters))
                break
        saver=tf.train.Saver()
        saver.save(sess,Config.newmodelpath,global_step=999)
        print("Model save!")

class Hit_helper():
    '''
    获取标定图片，预处理，并且截取出眼部图像
    '''
    def __init__(self):
        self.interface_block=int(Config.OUTPUTCLASS**0.5)
        self.data_helper=interface.Cal_Interface(line_num=self.interface_block,
                                            save_filename=Config.imagefile,
                                            wait_sec=1000,frame_num=20,time_gap=300,
                                            CAM_NUM=Config.CAM_NUM,
                                            sre_resolution=(Config.SRC_H,Config.SRC_W,3),
                                            frontalface_cascade=Config.frontalface_cascade,
                                            eye_cascade=Config.eye_cascade)
        self.face_cascade=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')# frontalface_cascade
        self.eye_cascade=cv2.CascadeClassifier('./haarcascade_eye.xml')#eye_cascade
    def star_cal(self):
        '''
        开始校准，获取图片
        :return:
        '''
        self.data_helper.starcalibrate()
    def process_data(self):
        '''
        对图片进行处理，获得眼部区域
        :return:
        '''
        return self.data_helper.imgprocess_eye()
    def geteyeimg(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3,5)
        if len(faces) !=1:

            return None
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            #检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
            eyes = self.eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.2, minNeighbors=7, minSize=(29, 29),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
            #眼睛检测 ,对于识别比较差的情况舍弃
            if len(eyes)!=2:
                return None
            if eyes[0][0]>eyes[1][0]:
                ex=eyes[1][0]
                W=eyes[0][0]-eyes[1][0]+eyes[0][2]
            else:
                ex=eyes[0][0]
                W=eyes[1][0]-eyes[0][0]+eyes[1][2]

            if eyes[0][1]>eyes[1][1]:
                ey=eyes[1][1]
                H=eyes[0][1]-eyes[1][1]+eyes[0][3]
            else:
                ey=eyes[0][1]
                H=eyes[1][1]-eyes[0][1]+eyes[1][3]

        return roi_color[ey+10:ey+H-2,ex-10:ex+W+10] if roi_color is not None else None

    def predict_RUNTEST(self,sess,PROB,X,bn_train,KEEP_PROB):
        '''
        预测函数
        此函数改成测试接口，用来测试一切功能正常，以及选取效果。
        这个函数去静态图片显示，而且只能选择一个区域
        :param sess:
        :param PROB:
        :param X:
        :param bn_train:
        :return:
        '''
        SRC_W,SRC_H=Config.SRC_W,Config.SRC_H
        target_img='medical2_big.jpg'
        fps=20.0
        fsize_desk=(SRC_W,SRC_H)
        prob=PROB
        video_d = cv2.VideoWriter(Config.Video_file,cv2.VideoWriter_fourcc(*'XVID'), fps, fsize_desk)

        cap=cv2.VideoCapture(Config.CAM_NUM)#眼部图像的摄像头
        try:
            # save video
            tar_img=cv2.imread(target_img)
            ori_img=cv2.resize(tar_img,(SRC_W,SRC_H))
            #设置window 为全屏
            cv2.namedWindow('Capture001',cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Capture001', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            #每N帧输出平均预测
            sum_score=[]
            sum_counter=0
            block_id=0
            #选中矩形框部分的变量
            change_flag=0
            block_counter={}
            rec_x,rec_y,rec_w,rec_h=0,0,0,0

            while True:
                ret,fram=cap.read()
                if ret:
                    #显示 内窥镜图像
                    tar_img=ori_img.copy()
                    tar_img=drawline(tar_img,wandh_num=self.interface_block)
                    cv2.imshow('Capture001',tar_img)
                    e=self.geteyeimg(fram)
                    if e is None:
                        continue
                    e=cv2.resize(e,(128,128))[:,:,::-1].reshape((1,128,128,3))
                    y_guess=sess.run(prob,feed_dict={X:e-Config.cumt_picmean,bn_train:False,KEEP_PROB:1.0})[0]

                    #平均每 N 帧的预测分数
                    sum_counter+=1
                    sum_score.append(y_guess)
                    if sum_counter==Config.MEAN_PICNUM:
                        sum_score=np.asarray(sum_score).reshape((-1,self.interface_block*self.interface_block))
                        mean_score=np.mean(sum_score,0)
                        block_id=np.argmax(mean_score)
                        sum_score=[]
                        sum_counter=0

                    #假如连续盯着一个地方，放大这部分,只放大两次
                    #block_id 表示这一帧图的预测值
                    if change_flag <2:
                        if block_id not in block_counter:
                            block_counter={}
                            block_counter[block_id]=0
                        block_counter[block_id]+=1
                        if block_counter[block_id]==30:#计数 共30*3=90帧图的预测值都相同
                            roi_rec,rec_shape=drawblock(tar_img,line_num=self.interface_block,block_id=block_id,show_rec='pick')
                            s_x,s_y,s_w,s_h=rec_shape#矩形区域的 (x,y,w,h)值
                            rec_x+=s_x//(self.interface_block**change_flag)
                            rec_y+=s_y//(self.interface_block**change_flag)
                            #print(rec_x,rec_y,s_x,s_y)
                            rec_w,rec_h=s_w//self.interface_block,s_h//self.interface_block
                            block_counter={}
                            ori_img=cv2.resize(roi_rec,(SRC_W,SRC_H))
                            change_flag+=1
                    elif change_flag ==2:
                        change_flag+=1
                        #将选中的区域标记处来
                        tar_img=cv2.imread(target_img)
                        ori_img=cv2.resize(tar_img,(SRC_W,SRC_H))
                        cv2.rectangle(ori_img,(rec_x,rec_y),(rec_x+rec_w,rec_y+rec_h),(0,0,255),3)
                        continue

                    tar_img=drawblock(tar_img,line_num=self.interface_block,block_id=block_id,show_rec='select')
                    v_img=tar_img.copy()
                    fram=cv2.resize(fram,(200,200))
                    v_img[:200,-200:]=fram
                    cv2.imshow('Capture001',tar_img)
                    video_d.write(v_img)
                    if cv2.waitKey(1)&0xff==27:
                        print('EXIT PROGRAM!')
                        break
        except Exception:
            pass
        video_d.release()
        cap.release()
        cv2.destroyAllWindows()

    def predict_RUN(self,sess,PROB,X,bn_train,KEEP_PROB):
        '''
        运行一次 进行两次缩放选取 得到一个矩形区域的位置信息
        可以运行两次这个程序获得 起始点、终止点的位置信息
        预测函数，CAM_0获得用户的眼部图像,CAM_1获得外部的环境图像，用于选取
        :param sess:
        :param PROB:
        :param X:
        :param bn_train:
        :param KEEP_PROB:
        :return:
        '''
        SRC_W,SRC_H=Config.SRC_W,Config.SRC_H
        fps=20.0
        fsize_desk=(SRC_W,SRC_H)
        prob=PROB
        cap=cv2.VideoCapture(Config.CAM_NUM)#眼部图像的摄像头
        cap_env=cv2.VideoCapture(Config.CAM_NUM_ENV)#环境图像的摄像头
        time_stamp=str(int(time.time()*1e7))
        video_d = cv2.VideoWriter("outputPick_"+time_stamp+".avi",cv2.VideoWriter_fourcc(*'XVID'), fps, fsize_desk)
        video_ori_desk = cv2.VideoWriter("outputPick_deskori_"+time_stamp+".avi",cv2.VideoWriter_fourcc(*'XVID'), fps,
                                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        video_ori_env = cv2.VideoWriter("outputPick_envori_"+time_stamp+".avi",cv2.VideoWriter_fourcc(*'XVID'), fps,
                                        (int(cap_env.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap_env.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        #Config.CAM_NUM_ENV 测试时两个相机都为脸部相机，方便调试

        try:
            # save video

            #设置window 为全屏
            cv2.namedWindow('Capture001',cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Capture001', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            #每N帧输出平均预测
            sum_score=[]
            sum_counter=0
            block_id=0
            #选中矩形框部分的变量
            change_flag=0
            block_counter={}
            rec_x,rec_y,rec_w,rec_h=0,0,0,0#保存目标矩形区域信息
            Cur_Blockid=-1
            while True:
                ret,fram=cap.read()#获取脸部图像
                ret2,fram_env=cap_env.read()#获取内窥镜图像
                if ret and ret2:
                    video_ori_desk.write(fram)#保存脸部图像，无修改
                    video_ori_env.write(fram_env)#保存内窥镜图像，无修改
                    ori_img=cv2.resize(fram_env,(SRC_W,SRC_H))#内窥镜图片
                    if (Cur_Blockid !=-1) and (Cur_Blockid !=-2):
                        ori_img,_=drawblock(ori_img,line_num=self.interface_block,block_id=Cur_Blockid,show_rec='pick')
                        ori_img=cv2.resize(ori_img,(SRC_W,SRC_H))
                    elif Cur_Blockid ==-2:
                        cv2.rectangle(ori_img,(rec_x,rec_y),(rec_x+rec_w,rec_y+rec_h),(0,0,255),3)
                    tar_img=ori_img.copy()
                    tar_img=drawline(tar_img,wandh_num=self.interface_block)
                    cv2.imshow('Capture001',tar_img)#显示 内窥镜图像（划线后）
                    e=self.geteyeimg(fram)#截取眼部区域
                    if e is None:#没有检测到眼部
                        continue
                    #预测
                    e=cv2.resize(e,(128,128))[:,:,::-1].reshape((1,128,128,3))
                    y_guess=sess.run(prob,feed_dict={X:e-Config.cumt_picmean,bn_train:False,KEEP_PROB:1.0})[0]

                    #平均每 N 帧的预测分数
                    sum_counter+=1
                    sum_score.append(y_guess)
                    if sum_counter==Config.MEAN_PICNUM:
                        sum_score=np.asarray(sum_score).reshape((-1,self.interface_block*self.interface_block))
                        mean_score=np.mean(sum_score,0)
                        block_id=np.argmax(mean_score)
                        sum_score=[]
                        sum_counter=0

                    #假如连续盯着一个地方，放大这部分,只放大两次
                    #block_id 表示这一帧图的预测值
                    if change_flag <2:
                        if block_id not in block_counter:
                            #block_counter={}
                            block_counter[block_id]=0
                        block_counter[block_id]+=1
                        if block_counter[block_id]==Config.WAITPIC_NUM:#计数 共30*3=90帧图的预测值都相同
                            roi_rec,rec_shape=drawblock(tar_img,line_num=self.interface_block,block_id=block_id,show_rec='pick')
                            s_x,s_y,s_w,s_h=rec_shape
                            rec_x+=s_x//(self.interface_block**change_flag)
                            rec_y+=s_y//(self.interface_block**change_flag)

                            rec_w,rec_h=s_w//self.interface_block,s_h//self.interface_block
                            #rec_x  rec_y  rec_w  rec_h 表示了当前选中的矩形区域的位置信息
                            block_counter={}
                            #ori_img=cv2.resize(roi_rec,(SRC_W,SRC_H))
                            change_flag+=1
                            Cur_Blockid=block_id
                    elif change_flag ==2:
                        change_flag+=1
                        #将选中的区域标记处来
                        #tar_img=cv2.imread(target_img)
                        #ori_img=cv2.resize(tar_img,(SRC_W,SRC_H))
                        cv2.rectangle(ori_img,(rec_x,rec_y),(rec_x+rec_w,rec_y+rec_h),(0,0,255),3)
                        #rec_x,rec_y,rec_w,rec_h=0,0,0,0
                        Cur_Blockid=-2
                        break

                    tar_img=drawblock(tar_img,line_num=self.interface_block,block_id=block_id,show_rec='select')
                    v_img=tar_img.copy()
                    fram=cv2.resize(fram,(200,200))
                    v_img[:200,-200:]=fram #将用户的图像也融合到环境里面去
                    cv2.imshow('Capture001',tar_img)
                    video_d.write(v_img)
                    if cv2.waitKey(1)&0xff==27:
                        print('EXIT PROGRAM!')
                        break
                else:
                    if not ret:
                        print('!'*80)
                        print("SOME THING WRONG WITH EYE CAM!!!!!")
                        print('!'*80)
                    if not ret2:
                        print('!'*80)
                        print("SOME THING WRONG WITH ENVIRONMENT CAM!!!!!")
                        print('!'*80)
        except Exception:
            print("Exception!!")
            pass
        video_d.release()
        video_ori_env.release()
        video_ori_desk.release()
        cap.release()
        cv2.destroyAllWindows()
        print("rec_x,rec_y,rec_w,rec_h:")
        print(rec_x,rec_y,rec_w,rec_h)
        return (rec_x,rec_y,rec_w,rec_h)

    def predict_RUN_DIRECTION(self,sess,PROB,X,bn_train,KEEP_PROB,info_corp):
        '''
        User Command模式，预测结果为 上下左右和保持
        分别对应的 BlockID为 ：
           1
        3  4  5
           7

        info_corp
        为辅助类，帮助发送信息
        :param sess:
        :param PROB:
        :param X:
        :param bn_train:
        :param KEEP_PROB:
        :return:
        '''
        SRC_W,SRC_H=Config.SRC_W,Config.SRC_H
        fps=20.0
        fsize_desk=(SRC_W,SRC_H)
        prob=PROB
        time_stamp=str(int(time.time()*1e7))

        cap=cv2.VideoCapture(Config.CAM_NUM)#眼部图像的摄像头
        cap_env=cv2.VideoCapture(Config.CAM_NUM_ENV)#环境图像的摄像头
        video_d = cv2.VideoWriter("outputCommand_"+time_stamp+".avi",cv2.VideoWriter_fourcc(*'XVID'), fps, fsize_desk)
        video_ori_desk = cv2.VideoWriter("outputCommand_deskori_"+time_stamp+".avi",
                                         cv2.VideoWriter_fourcc(*'XVID'), fps,
                                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        video_ori_env = cv2.VideoWriter("outputCommand_envori_"+time_stamp+".avi",cv2.VideoWriter_fourcc(*'XVID'), fps,
                                        (int(cap_env.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap_env.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        #Config.CAM_NUM_ENV 测试时两个相机都为脸部相机，方便调试
        try:
            # save video

            #设置window 为全屏
            cv2.namedWindow('Capture001',cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Capture001', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            #每N帧输出平均预测
            sum_score=[]
            sum_counter=0
            block_id=0
            #选中矩形框部分的变量
            change_flag=0
            block_counter={}
            rec_x,rec_y,rec_w,rec_h=0,0,0,0#保存目标矩形区域信息
            Cur_Blockid=-1
            Pre_Blockid=-1

            while True:
                ret,fram_eye=cap.read()#获取脸部图像
                ret2,fram_env=cap_env.read()#获取内窥镜图像
                if ret and ret2:
                    video_ori_desk.write(fram_eye)#保存脸部图像，无修改
                    video_ori_env.write(fram_env)#保存内窥镜图像，无修改
                    ori_img=cv2.resize(fram_env,(SRC_W,SRC_H))#内窥镜图片
                    if Cur_Blockid in {1,3,4,5,7,-2}:
                        #将这个区域用其他颜色高亮出来
                        ori_img=drawblock(ori_img,line_num=self.interface_block,block_id=Cur_Blockid,show_rec='fill',blockcolor=(46,218,255))
                        if Pre_Blockid ==-1:
                            Pre_Blockid=Cur_Blockid
                        if Cur_Blockid==-2:#将这个区域写入文件,发送出去。并将状态回复
                            print("Send ALL!!")
                            info_corp.writeinfo(block_id,infotype='command')
                            print("sending.....")
                            info_corp.sendAll()
                            Pre_Blockid=-1
                            Cur_Blockid=-1
                            change_flag=0
                            block_counter={}
                            rec_x,rec_y,rec_w,rec_h=0,0,0,0


                    tar_img=ori_img.copy()
                    tar_img=drawline(tar_img,wandh_num=self.interface_block)
                    cv2.imshow('Capture001',tar_img)#显示 内窥镜图像（划线后）
                    e=self.geteyeimg(fram_eye)#截取眼部区域
                    if e is None:#没有检测到眼部
                        continue
                    #预测
                    e=cv2.resize(e,(128,128))[:,:,::-1].reshape((1,128,128,3))
                    y_guess=sess.run(prob,feed_dict={X:e-Config.cumt_picmean,bn_train:False,KEEP_PROB:1.0})[0]

                    #平均每 N 帧的预测分数
                    sum_counter+=1
                    sum_score.append(y_guess)
                    if sum_counter==Config.MEAN_PICNUM:
                        sum_score=np.asarray(sum_score).reshape((-1,self.interface_block*self.interface_block))
                        mean_score=np.mean(sum_score,0)
                        block_id=np.argmax(mean_score)
                        sum_score=[]
                        sum_counter=0

                    #假如连续盯着一个地方，放大这部分,只放大两次
                    #block_id 表示这一帧图的预测值
                    if change_flag <2:
                        if block_id not in block_counter:
                            #block_counter={}
                            block_counter[block_id]=0
                        block_counter[block_id]+=1
                        if block_counter[block_id]==Config.WAITPIC_NUM:#计数 共30*3=90帧图的预测值都相同
                            roi_rec,rec_shape=drawblock(tar_img,line_num=self.interface_block,block_id=block_id,show_rec='pick')
                            s_x,s_y,s_w,s_h=rec_shape
                            rec_x+=s_x//(self.interface_block**change_flag)
                            rec_y+=s_y//(self.interface_block**change_flag)

                            rec_w,rec_h=s_w//self.interface_block,s_h//self.interface_block
                            #rec_x  rec_y  rec_w  rec_h 表示了当前选中的矩形区域的位置信息
                            block_counter={}
                            #ori_img=cv2.resize(roi_rec,(SRC_W,SRC_H))
                            change_flag+=1
                            Cur_Blockid=block_id
                    elif change_flag ==2:
                        if Cur_Blockid==Pre_Blockid:#第二次选择时，必须和上次选择是同样的结果
                            change_flag+=1
                            Cur_Blockid=-2
                        else:#否则将会清除前一个的选择，重新选区域
                            print("Clear ALL!!")
                            Cur_Blockid=-1
                            change_flag=0
                            block_counter={}
                            rec_x,rec_y,rec_w,rec_h=0,0,0,0
                            info_corp.cleanInfoPool()
                            Pre_Blockid=-1

                        #continue

                    tar_img=drawblock(tar_img,line_num=self.interface_block,block_id=block_id,show_rec='select')
                    v_img=tar_img.copy()
                    fram=cv2.resize(fram_eye,(200,200))
                    v_img[:200,-200:]=fram #将用户的图像也融合到环境里面去
                    cv2.imshow('Capture001',tar_img)
                    video_d.write(v_img)
                    if cv2.waitKey(1)&0xff==27:
                        print('EXIT PROGRAM!')
                        break
                else:
                    if not ret:
                        print('!'*80)
                        print("SOME THING WRONG WITH EYE CAM!!!!!")
                        print('!'*80)
                    if not ret2:
                        print('!'*80)
                        print("SOME THING WRONG WITH ENVIRONMENT CAM!!!!!")
                        print('!'*80)
        except Exception:
            print("Exception!!")
            pass
        video_d.release()
        video_ori_env.release()
        video_ori_desk.release()
        cap.release()
        cv2.destroyAllWindows()

def drawline(img_,line_w=1,line_color=(0,0,0),wandh_num=4):

    '''
    在图片上格子
    :param line_w: 线宽
    :param line_color: 线颜色
    :param wandh_num:  长宽线的数量
    :return:  无
    '''
    h,w=img_.shape[0],img_.shape[1]
    w_num,h_num=wandh_num,wandh_num
    h_,w_=h//h_num,w//w_num

    # 竖线 (w,h)
    for i in range(1,w_num):
        #print(i)
        cv2.line(img_,(w_*i,0),(w_*i,h),line_color,line_w)
    # 横线
    for i in range(1,h_num):
        cv2.line(img_,(0,h_*i),(w,h_*i),line_color,line_w)
    return img_
def drawblock(img,line_num,block_id=0,blockcolor=(46,218,255),blockwideth=5,show_rec='fill',rec_shape=None):
    '''
    选定九宫格，在这个格子上填充矩形表示选定这个格子
    :param img_: 图片
    :param block: 九宫格序号 0-15
    :param blockcolor: 矩形框颜色
    :param blockwideth: 框的宽度
    :return:
    '''
    h,w=img.shape[0],img.shape[1]
    w_line,h_line=line_num,line_num
    h_,w_=h//h_line,w//w_line
    cor_h=block_id//line_num
    cor_w=block_id%line_num
    sx,sy=cor_w*w_,cor_h*h_

    if show_rec=='fill':
        #将整个矩形填充为其他颜色
        img[sy:sy+h_,sx:sx+w_,:]=blockcolor
    elif show_rec=='rec':
        #显示矩形轮廓
        xe,ye,we,he=rec_shape
        cv2.rectangle(img,(xe,ye),(xe+we,ye+he),(255,0,0),10)
    elif show_rec=='dot':
        #标记一个小点
        roi_=img[sy:sy+h_,sx:sx+w_]
        cv2.circle(roi_,(roi_.shape[1]//2,roi_.shape[0]//2), 10, (255,128,120), -1)
    elif show_rec=='select':
        #hight light 矩形区域
        #img[sy:sy+h_,sx:sx+w_,0]=255#r
        img[sy:sy+h_,sx:sx+w_,1]=255#g
        #img[sy:sy+h_,sx:sx+w_,2]=255#b
    elif show_rec=='pick':
        #返回矩形框内容
        return img[sy:sy+h_,sx:sx+w_],(sx,sy,w_,h_)
    return img

def Mode_PickPlace():
    '''
    论文中的 Pick and Place 模式
    选中起始区域，目标区域，将坐标发送出去
    :return:
    '''
    data_helper,model_dense166,infoCorp=CalibrationModel()

    start_point=data_helper.predict_RUN(model_dense166.sess,model_dense166.prob,model_dense166.X,model_dense166.bn_train,model_dense166.KEEP_PROB)
    end_point=data_helper.predict_RUN(model_dense166.sess,model_dense166.prob,model_dense166.X,model_dense166.bn_train,model_dense166.KEEP_PROB)
    #发送起始点 终止点信息给 服务器
    print('*'*25+"Send to Server"+'*'*25)
    infoCorp.writeinfo(start_point,infotype='pick')
    infoCorp.writeinfo(end_point,infotype='pick')
    infoCorp.sendAll()


def Mode_Command():
    '''
    论文中的 Pick and Place 模式
    选中起始区域，目标区域，将坐标发送出去
    :return:
    '''
    data_helper,model_dense166,infoCorp=CalibrationModel()
    data_helper.predict_RUN_DIRECTION(model_dense166.sess,model_dense166.prob,model_dense166.X,model_dense166.bn_train,model_dense166.KEEP_PROB,infoCorp)

def CalibrationModel():
    '''
    校正模型,通过这个函数来初始化模型，校正模型
    :return:
    '''
    #加载数据辅助类，主要用于获取校准图片
    data_helper=Hit_helper()
    #DenseNet 模型类
    model_dense166=Hit_traker()
    #通信模块,将坐标或者方向信息通过这个接口写入。再发送出去
    infoCorp=signalcorps(HOST=Config.HOST,PORT=Config.PORT)
    data_helper.data_helper.usercheck(Config.CAM_NUM)
    if Config.RUN_TRAIN:#是否训练模型
        if Config.RUN_CAL:#是否要运行程序获得校准图片（最好不要通过这个方式，而是通过interface来获得图片）
            print('Start calibration,getting image!')
            data_helper.star_cal()
        eye_data,eye_label=data_helper.process_data()
        print('Trainning model,it may take a while!!')
        #训练模型
        model_dense166.finetuneNetwork(eye_data,eye_label)
    return data_helper,model_dense166,infoCorp
def GetMoreData():
    '''
    利用这个接口函数获得更多的训练图片
    '''
     #加载数据辅助类，主要用于获取校准图片
    data_helper=Hit_helper()
    print('Start calibration,getting image!')
    data_helper.star_cal()
def processimage(imgfile,pkl_output):
    '''
    将图像文件转换成眼部的pkl文件
    :param imgfile:
    :return:
    '''
    data_helper=Hit_helper()
    print('Converting image to pickle file.....')
    eye_data,eye_label=data_helper.data_helper.imgprocess_eye(imagefile=imgfile)
    with open(pkl_output,'wb') as fp:
        pickle.dump(obj={'data':eye_data,'labels':eye_label},
                    file=fp,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("Convert Done.Save pickle file:{}".format(pkl_output))

def localtest():
    '''
    用于测试CNN模型的实际运行效果，主要区别在于去掉了外部通信
    测试模式为 UserCommand
    :return:
    '''
    print("本地测试中.....")
    #加载数据辅助类，主要用于获取校准图片
    data_helper=Hit_helper()
    #DenseNet 模型类
    model_dense166=Hit_traker()
    #通信模块,将坐标或者方向信息通过这个接口写入。再发送出去
    infoCorp=signalcorps_localtest()
    data_helper.data_helper.usercheck(Config.CAM_NUM)
    data_helper.predict_RUN_DIRECTION(model_dense166.sess,model_dense166.prob,model_dense166.X,model_dense166.bn_train,model_dense166.KEEP_PROB,infoCorp)
if __name__ == '__main__':
    #pick and place 模式
    #Mode_PickPlace()

    #User command 模式
    #Mode_Command()

    #单独运行，获取更多的图片数据
    #GetMoreData()

    #单独运行，将图片转换成pkl文件
    #processimage(imgfile,pkl_output)

    #测试模型表现，去除外部通信
    localtest()
