import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import cv2
from cumt_variant_fc import densenet
import os

model_path='model_0507/Mingle/Dense22_3X3/Dense22_new-0'
newmodel_path='model_0507/Dense22_3X3/newmodel/Dense22_finetune'
batchsize=32
lr_=8e-4
epoch_iters=10000
cumt_picmean=[103.939, 116.779, 123.68]
sess=tf.InteractiveSession()
init=tf.global_variables_initializer()
#Dense22 K=12,L=2,theta=0.5,block_num=4
#修改最后的AVG-POOL 为 FC层
def prep_data_augment(image):
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    return image
def data_augment(input_tensor):
    output_tensor = tf.map_fn(prep_data_augment, input_tensor)
    return output_tensor
output_class=9 #实际数据

with tf.variable_scope('Placehloder'):
    X=tf.placeholder(dtype=tf.float32,shape=[None,128,128,3],name='X')
    Y=tf.placeholder(dtype=tf.float32,shape=[None,output_class],name='Y')
    bn_train=tf.placeholder(dtype=tf.bool,name='BN_FLAG')
    LR=tf.placeholder(dtype=tf.float32,name='lr')
    KEEP_PROB=tf.placeholder(dtype=tf.float32,name='KEEP_PROB')
with tf.name_scope('DenseNet'):
    X_aug=data_augment(X)
    model=densenet(Images=X_aug,bn_istraining=bn_train,K=12,L=2,theta=0.8,output_class=output_class,sess=sess,denseblock_num=4,KEEP_PROB=KEEP_PROB)
    y_score=model.y_score
with tf.name_scope('LOSS'):
    LOSS=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_score,labels=Y))
    reg_LOSS=model.get_l2loss()
    LOSS+=1e-4*reg_LOSS
    tf.summary.scalar('loss',LOSS)
with tf.name_scope('TRAIN'):
    TRAIN=tf.train.AdamOptimizer(LR).minimize(LOSS)
    TRAIN2=tf.train.GradientDescentOptimizer(LR).minimize(LOSS)
    #TRAIN=tf.train.MomentumOptimizer(LR).minimize(LOSS)
with tf.name_scope('ACCURACY'):
    acc_count=tf.equal(tf.argmax(y_score,1),tf.argmax(Y,1))
    ACCURACY=tf.reduce_mean(tf.cast(acc_count,tf.float32))
    tf.summary.scalar('acc',ACCURACY)
bn_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

d=np.load('fine2.pkl')

index_=np.arange(d['data'].shape[0])
np.random.shuffle(index_)
tr_index=index_[:int(d['data'].shape[0]*0.8)]
te_index=index_[int(d['data'].shape[0]*0.8):int(d['data'].shape[0]*0.9)]
val_index=index_[int(d['data'].shape[0]*0.9):]
tr_data,tr_label=d['data'][tr_index],d['labels'][tr_index]
te_data,te_label=d['data'][te_index],d['labels'][te_index]
val_data,val_label=d['data'][val_index],d['labels'][val_index]
print("train data size:{},test data size:{},val data size:{}".format(tr_data.shape[0],te_data.shape[0],val_data.shape[0]))
del d
tr_data=np.float32(tr_data-cumt_picmean)
te_data=np.float32(te_data-cumt_picmean)
saver=tf.train.Saver()
saver.restore(sess,model_path)
#开始训练
print('*'*50)
print('START TRAINNING')


overfit_count=0
test_count=0
kp_prob=0.3
test_batch=min(128,tr_data.shape[0])
now_train=TRAIN2
for i in range(1,epoch_iters):
    mask=np.random.choice(tr_data.shape[0],batchsize,replace=False)
    x_,y_=tr_data[mask],tr_label[mask]
    feed_dict={X:x_,Y:y_,bn_train:True,LR:lr_,KEEP_PROB:kp_prob}
    sess.run([now_train,bn_ops],feed_dict=feed_dict)
    if i%20==0:
        mask=np.random.choice(tr_data.shape[0],batchsize*2,replace=False)#
        x_,y_=tr_data[mask],tr_label[mask]
        feed_dict={X_aug:x_,Y:y_,bn_train:False,KEEP_PROB:1.0}
        loss_,acc_=sess.run([LOSS,ACCURACY],feed_dict=feed_dict)
        print('epoch:{},train loss:{},train accuracy:{}'.format(i,loss_,acc_))
        if acc_>=0.999:
            print ('!'*10)
            print(overfit_count)
            overfit_count+=1
    if i%10==0:
        mask=np.random.choice(te_data.shape[0],test_batch,replace=True)
        x_,y_=te_data[mask],te_label[mask]
        feed_dict={X_aug:x_,Y:y_,bn_train:False,KEEP_PROB:1.0}
        loss_,acc_=sess.run([LOSS,ACCURACY],feed_dict=feed_dict)
        print('--epoch:{},test loss:{},test accuracy:{}'.format(i,loss_,acc_))
        if acc_>=0.90:
            print ('!'*10+'good result'+'!'*10)
            print(test_count)
            test_count+=1
    if overfit_count>50 or test_count>20:
        print ('!'*80)
        print('Early stop')
        print('overfit count {}/50, test_good count {}/5'.format(overfit_count,test_count))
        print ('!'*80)
        break

#在验证集上验证准确率
print('#'*50)
print('START EVALING')
test_batch=min(128,val_data.shape[0])
loop_=val_data.shape[0]//test_batch
idx_=0
acc_lis=[]
y_pred,y_true=[],[]
Y_PROB=tf.nn.softmax(y_score)
Y_PRE=tf.argmax(Y_PROB,axis=1)
for j in range(loop_):
    x_,y_=val_data[idx_:idx_+test_batch],val_label[idx_:idx_+test_batch]
    feed_dict={X_aug:x_-cumt_picmean,Y:y_,bn_train:False,KEEP_PROB:1.0}
    loss_,acc_,p=sess.run([LOSS,ACCURACY,Y_PRE],feed_dict=feed_dict)
    y_pred.extend(list(p))
    y_true.extend(list(y_))
    acc_lis.append(acc_)
    idx_+=test_batch
runlog_='data size:{},trans accuracy:{} \n'.format(val_data.shape[0],np.mean(acc_lis))
print(runlog_)
saver=tf.train.Saver()
saver.save(sess,newmodel_path,global_step=0)
print("ALL DONE!")

