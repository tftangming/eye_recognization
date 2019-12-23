#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-22 下午1:52
# @Author  : tangming
'''
说明：
    class Cal_Interface 希望有的功能：
        将屏幕分割成小方格，随机将一定数量的小方格标记出来
        当用户在注视这些小方格的时候用摄像头捕获用户图像，保存下来
        作为标定图片

    2017年10月16日：
        完成：
            1.每隔 1s 标识出一个矩形框

            2.在摄像头中捕获用户的图像，并且保存

'''
import cv2
import numpy as np
import time
import os
from Config import Config

class Cal_Interface(object):
    def __init__(self, sre_resolution=(1080, 1920, 3), cal_time=None, wait_sec=1000,
                 time_gap=200, frame_num=3, save_filename='calimg_file',
                 line_num=10, line_width=1, line_color=(255, 255, 255), blockcolor=(250, 180, 222), blockwideth=5,
                 frontalface_cascade=
                 './haarcascade_frontalface_default.xml',
                 eye_cascade='./haarcascade_eye.xml',
                 CAM_NUM=0):
        '''
         
        初始化函数
        :param sre_resolution: 屏幕分辨率，默认为 （1080,1920,3）
        :param cal_time: 校正点的数量
        :param wait_sec: 当点显示后延时多久开始捕捉图像
        :param time_gap:每一帧图片之间的时间间隔
        :param frame_num:每个点捕捉的图像帧数
        :param line_num:  图像画的横、竖线个数
        :param line_width:  线宽
        :param line_color:  线的颜色
        :param blockcolor:  矩形框的颜色
        :param blockwideth:  矩形框的宽度
        :param frontalface_cascade:  用于脸部识别的文件地址
        :param eye_cascade:  用于眼部识别的文件地址
        :param CAM_NUM:  默认摄像头为 0
        '''
        # 默认分辨率为 （1080,1920）
        self.img_res=sre_resolution
        # 默认摄像头为 0
        self.CAM_NUM=CAM_NUM
        # 保存文件夹位置
        if not os.path.exists(save_filename):
            os.mkdir(save_filename)
            print('Create image file:{}'.format(save_filename))
        self.save_filename=save_filename
        # 校正点的个数
        if cal_time is None:
            self.caltime=line_num*line_num
        else:
            self.caltime=cal_time
        # 延时多久开始捕捉图像
        self.wait_sec=wait_sec
        # 每一帧图片之间的时间间隔
        self.time_gap=time_gap
        # 每个点捕捉的图像帧数
        self.framenum=frame_num-1
        # 保存画线的参数
        self.line_num=line_num  #线的数量
        self.line_width=line_width  #线宽
        self.line_color=line_color  #线颜色
        # 保存方框格参数
        self.blockwideth=blockwideth  #线宽
        self.blockcolor=blockcolor  #线颜色
        

        # OpenCV脸部识别文件
        self.face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')# frontalface_cascade
        self.eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')#eye_cascade


    def display_grid(self):
        '''
        @@@废弃，没有在类中调用@@@
        
        全屏显示图片,等待键盘输入任意值摧毁窗口
        :return:
        '''
        img_=self.cal_img
        cv2.namedWindow('Calibrate',cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibrate', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Calibrate',img_)
        cv2.waitKey()
        cv2.destroyWindow('Calibrate')

    def drawline(self, img_, line_w=1, line_color=(0, 0, 0), wandh_num=3):
        '''
        在图片上格子
        :param line_w: 线宽
        :param line_color: 线颜色
        :param wandh_num:  长宽线的数量
        :return:  无
        '''
        h, w = img_.shape[0], img_.shape[1]
        w_num, h_num = wandh_num, wandh_num
        h_, w_ = h//h_num,w//w_num

        # 竖线 (w,h)
        for i in range(1,w_num):
            #print(i)
            cv2.line(img_, (w_*i, 0), (w_*i, h), line_color, line_w)
        # 横线
        for i in range(1,h_num):
            cv2.line(img_, (0, h_*i), (w, h_*i), line_color, line_w)

    def drawblock(self,img,block_id=0,blockcolor=(128,128,128),blockwideth=5):
        '''
        选定九宫格，在这个格子上填充矩形表示选定这个格子
        :param img_: 图片
        :param block: 九宫格序号
        :param blockcolor: 矩形框颜色
        :param blockwideth: 框的宽度
        :return:
        '''
        # blockcolor=(218,218,218)
        h, w = img.shape[0], img.shape[1]
        w_line, h_line = self.line_num, self.line_num
        h_, w_ = h//h_line, w//w_line
        cor_h = block_id//self.line_num
        cor_w = block_id % self.line_num
        sx, sy = cor_w*w_, cor_h*h_

        # 将整个矩形填充为其他颜色
        img[sy:sy+h_, sx:sx+w_, :] = blockcolor
        # 在矩形中心画一个小圆辅助
        roi_ = img[sy:sy+h_, sx:sx+w_]
        random_w = np.random.choice(range(20, w_-20),1)[0]
        random_h = np.random.choice(range(20, h_-20),1)[0]
        # cv2.circle(roi_,(roi_.shape[1]//2,roi_.shape[0]//2), 50, (0,0,0), -1)
        cv2.circle(roi_, (random_w, random_h), 20, (0, 0, 0), -1)

        # 只是在矩形边缘画框
        # cv2.rectangle(img_,(sx,sy),(sx+w_,sy+h_),blockcolor,blockwideth)
        return img

    def cal(self,cal_times,wait_sec=1000):
        '''
        校准函数，随机显示若干个方格，捕获用户的图片，保存下来
        按 Esc 退出
        修改lable 为block id，也就是改预测模型为分类而不是回归
        不要设置 cal_times 参数，不能起作用
        :param cal_times: 显示的方格个数
        :param wait_sec: 每个方格出现的时间间隙
        :return:
        '''
        # 这里可以创建乱序的block id
        if Config.JUST_DOWN:
            block_index=np.array(range(30, 36))
        else:
            block_index=np.arange(self.line_num*self.line_num)
        # np.random.shuffle(block_index)

        # 设置window 为全屏
        cv2.namedWindow('Calibrate', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibrate', cv2.WND_PROP_FULLSCREEN, cv2.CV_WINDOW_AUTOSIZE)
        # 随机显示某个位置的方格，设置1 s后保存得到的图像
        for index_ in block_index:
            print(index_)
            img_ = cv2.imread('test.jpg')
            self.drawline(img_=img_, wandh_num=self.line_num,line_color=(255,255,255))
            self.drawblock(img=img_, block_id=index_, blockcolor=self.blockcolor)
            cv2.imshow('Calibrate', img_)
            # 等待 1s 后保存图像 wait_sec
            if cv2.waitKey() & 0xff == 27:
                print('Exit Calibrate!')
                break
            # 保存用户图像
            self.GetUserImage(savefile=self.save_filename, label=index_, frame_num=self.framenum, block_index=index_)
        cv2.destroyWindow('Calibrate')
        print('Get Calibrate Image Done!')

    def GetUserImage(self,block_index,label=None,savefile='calimg_file',frame_num=10):
        '''
        获取用户图像并且保存
        一次调用 存储frame_num张图片，每张图片之间的时间间隙为frame_gap
        一共耗时 frame_gap*frame_num 毫秒
        :param label: 当前用户注视的坐标方向
        :param savefile: 保存路径
        :param frame_num: 每次坐标捕捉的帧数
        :return: 无
        '''
        cap=self.cap
        cap2=self.cap2
        s_time = time.time()
        f_counter = 0
        time_gap = self.time_gap
        # 捕捉帧
        while True:
            img_ = cv2.imread('test.jpg')
            self.drawline(img_=img_, wandh_num=self.line_num, line_color=(255, 255, 255))
            self.drawblock(img=img_, block_id=block_index, blockcolor=self.blockcolor)
            cv2.imshow('Calibrate', img_)
            cv2.waitKey(time_gap)
            ret, fram = cap.read()
            ret2 = 0
            fram2 = None
            if cap2 is not None:
                ret2, fram2 = cap2.read()
            if ret:
                f_counter+=1
                time_stamp=str(int(time.time()*1e7))
                # block id
                img_name=time_stamp+'_'+'blockid_'+str(label)+'.jpg'
                cv2.imwrite(os.path.join(savefile,img_name),fram)
                if ret2:
                    time_stamp+='b'
                    #block id
                    img_name=time_stamp+'_'+'blockid_'+str(label)+'.jpg'
                    cv2.imwrite(os.path.join(savefile,img_name),fram2)
            # 30秒 超时退出
            if (time.time()-s_time>30):
                print('Some thing wrong with the cam,time out!')
                break
            if (f_counter>frame_num):
                #print('get user picture ok,save picture:{}'.format(f_counter))
                break

    def cameracheck(self):
        '''
        标定程序运行时的视角检查，会显示用户图像并且标记出眼睛、脸
        用来检查当前相机姿态，用户姿态是否合适
        按 ESC 退出或者是 1分钟后超时退出
        :return: 
        '''
        
        face_cascade=self.face_cascade
        eye_cascade=self.eye_cascade
        cap=self.cap
        s_time=time.time()
        print('start camera check!')
        while True:
            ret, frame = cap.read()
            if ret:
                self.drew_face_eye(frame)
                cv2.imshow('test', frame)
                if cv2.waitKey(30)&0xff == 27:
                    print('camera check ok!')
                    break
            rt=time.time()-s_time
            # 超时退出
            if rt >60*10:
                print('camera check time out')
                break
        if self.cap2 is not None:
            while True :
                cap = self.cap2
                ret,frame=cap.read()
                if ret:
                    self.drew_face_eye(frame)
                    cv2.imshow('test',frame)
                    if cv2.waitKey(30)&0xff == 27:
                        print('camera check ok!')
                        break
                rt=time.time()-s_time
                #超时退出
                if rt >60*10:
                    print('camera check time out')
                    break
        cv2.destroyWindow('test')

    def drew_face_eye(self,img):
        '''
        检测用户当前环境是否能够比较好的识别面部特征
        辨识出脸部以及眼部，标记出来
        :param img:
        :return:
        '''
        face_cascade = self.face_cascade
        eye_cascade = self.eye_cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            # 检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=7, minSize=(29, 29),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            for e in eyes:
                xe, ye, we, he = e
                cv2.rectangle(roi_color, (xe, ye), (xe+we, ye+he), (0, 0, 255), 2)
        return img

    def getoutclean(self):
        '''
        在结束标定的时候进行清理
        :return:
        '''
        self.cap.release()
        if self.cap2 is not None:
            self.cap2.release()
        print('cam release!')
        cv2.destroyAllWindows()
        print('all windows destroy!')

    def showpicnum(self):
        '''
        返回保存照片文件夹的图片地址
        用来显示当前图片数量
        :return:
        '''
        file_addr=self.save_filename
        return os.listdir(file_addr)

    def starcalibrate(self):
        # 摄像头配置
        self.cap = cv2.VideoCapture(self.CAM_NUM)
        self.cap2 = None
        if Config.CAM_2:
            print("Use Two Cam!!!")
            self.cap2 = cv2.VideoCapture(Config.CAM_2)
        # 检查用户当前位置是否可以识别出人脸和人眼
        self.cameracheck()
        #
        self.cal(self.caltime,wait_sec=self.wait_sec)
        self.getoutclean()
        print('Picture number:{}'.format(len(self.showpicnum())))

    def usercheck(self,CAM_NUM):
        '''
        用户摄像头自检查
        '''
        self.cap=cv2.VideoCapture(CAM_NUM)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS,100)
        self.cap2=None
        self.cameracheck()
        self.cap.release()
        print('cam release!')
        cv2.destroyAllWindows()
        print('all windows destroy!')

    def geteyeimg(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = self.face_cascade.detectMultiScale(gray, 1.3,5) 
        if len(faces) !=1:
            #print(len(faces))
            #print('bad faces')
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
                    #print(len(eyes))
                    #print('bad eyes')
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
    def imgprocess_eye(self,imagefile=None):
        '''
        截取出校正原始图片中的眼部图像
        并根据文件名打上标签
        :return:
        '''
        if imagefile is None:
            imagefile=self.save_filename
        img_dir=imagefile
        eye_lis,label_lis=[],[]
        fail_counter=0
        c=0
        for addr_ in os.listdir(img_dir):
            c+=1
            label_id=int(addr_[:-4].split('_')[-1])
            label_=[0]*Config.OUTPUTCLASS
            label_[label_id]=1
            img_=cv2.imread(os.path.join(img_dir,addr_))
            eyes_=self.geteyeimg(img_)
            if eyes_ is None:
                fail_counter+=1
                #print(fail_counter)
                continue
            eyes_=cv2.resize(eyes_,(128,128))
            eye_lis.append(eyes_[:,:,::-1])
            label_lis.append(label_)
        eye_lis=np.array(eye_lis)
        label_lis=np.array(label_lis)
        #print('Total image num:{},fail count:{} !'.format(len(list(os.listdir(img_dir)))),c)
        print(eye_lis.shape,label_lis.shape)
        return eye_lis,label_lis
if __name__ == '__main__':

    a=Cal_Interface(line_num=6,
                    save_filename='./img6x6_finetune_laptop_add',
                    wait_sec=1000, frame_num=20, time_gap=300,
                    CAM_NUM=0,
                    sre_resolution=(1366,768,3),
                    frontalface_cascade='haarcascade_frontalface_default.xml',
                    eye_cascade='haarcascade_eye.xml')
    a.starcalibrate()
