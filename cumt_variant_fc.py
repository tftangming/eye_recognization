

















import tensorflow as tf
from cnn_basic import bisic_cnn


class densenet(bisic_cnn):
    def __init__(self,Images,theta,K,L,bn_istraining,sess,denseblock_num,KEEP_PROB,output_class=10):
        self.theta=theta
        self.K=K
        self.L=L
        self.bn_istraining=bn_istraining
        self.output_class=output_class
        self.denseblock_num=denseblock_num
        self.KEEP_PROB=KEEP_PROB
        print('Building DenseNet....')
        self._build_densenet(Images)

    def _build_densenet(self,X):
        '''
        建立DenseNet网络:
            head:
                (N,128,128,3) -> (N,64,64,24)
            DenseConect:
                0:(N,64,64,24) -> (N,64,64,84) -> (N,32,32,42)
                1:(N,32,32,42) -> (N,32,32,102) -> (N,16,16,51)
                2:(N,16,16,51) -> (N,16,16,111) -> (N,8,8,55)
                3:(N,8,8,55) -> (N,8,8,115) -> (N,4,4,57)
            Tail:
                CONV_class:(N,4,4,57) ->(N,4,4,10)
                avg_pool:(N,4,4,10) -> (N,1,1,10)
                softmax:(N,1,1,10) -> (N,10)
        :param X:
        :param bn_istraing:
        :return:
        '''
        K=self.K
        L=self.L
        theta=self.theta
        bn_istraining=self.bn_istraining
        #(N,H,W,3) -> (N,H/2,W/2,2*K)
        #(N,128,128,3) -> (N,64,64,24)
        with tf.variable_scope('Head'):
            conv1_=self._convblock(x=X,name='conv1_3X3',shape=[3,3,3,2*K],bn_istraing=bn_istraining,padding='SAME',stride=[1,1],mode='conv_bn_relu')
            #conv2_=self._convblock(x=conv1_,name='conv2_3X3',shape=[3,3,K,2*K],bn_istraing=bn_istraining,padding='SAME',stride=[1,1],mode='conv_bn_relu')
            pool1_=self._poollayer(input_x=conv1_,pooling='max',size=(3,3),stride=(2,2),padding='SAME')
        print(pool1_.get_shape())
        with tf.variable_scope('DenseConect'):
            dense_input=pool1_
            for block_id in range(self.denseblock_num):
                block_name_dense='denseblock'+'_'+str(block_id+1)
                block_name_trans='transblock'+'_'+str(block_id+1)
                trans_input=self.dense_block(input_x=dense_input,name=block_name_dense,growth_tate=K,L=L)
                print(trans_input.get_shape())
                dense_input=self.trans_block(input_x=trans_input,name=block_name_trans,theta=theta)
                print(dense_input.get_shape())
        with tf.variable_scope('Tail'):
            #(N,4,4,57) ->(N,4,4,10)
            #dense_input=tf.nn.dropout(dense_input,self.dropout,name='drop_1')
            conv_class_=self._convblock(x=dense_input,name='conv_class',shape=[1,1,int(dense_input.get_shape()[-1]),self.output_class],bn_istraing=bn_istraining,padding='VALID',stride=[1,1])
            #(N,4,4,10) ->(N,1,1,10)
            dim_=1
            for d_ in conv_class_.get_shape().as_list()[1:]:
                dim_*=d_
            x_flatten=tf.reshape(conv_class_,[-1,dim_])
            x_flatten=tf.nn.dropout(x_flatten,self.KEEP_PROB)
            fc1=tf.nn.relu(self.fc_layer(x_flatten,'fc1',shape=[dim_,256]))
            fc1=tf.nn.dropout(fc1,self.KEEP_PROB)
            y_score=self.fc_layer(fc1,'fc2',shape=[256,self.output_class])
        self.y_score=y_score

    def trans_block(self,input_x,name,theta=0.5):
        '''
        连接两个Denseblock的中间环节，主要作用：1X1 CONV 缩减C -> 2X2 avgpool 缩减 H,W
        输入 C0=K0+K*L，输出C1=0.5*C0
        :param input_x: 输入
        :param theta: 缩减C的幅度
        :param bn_istraning: BN训练标志位
        :param name: 层名称
        :return:[N,H/2,W/2,C/2]
        '''
        bn_istraining=self.bn_istraining
        input_depth=int(input_x.get_shape()[-1])
        #1X1 conv -> bn -> relu，缩减C
        bottle_x=self._convblock(x=input_x,name=name+r'/1X1conv',shape=[1,1,input_depth,int(theta*input_depth)],bn_istraing=bn_istraining,padding='VALID',stride=[1,1])
        #2X2 avgpool，缩减H,W
        avg_pool_=self._poollayer(input_x=bottle_x,pooling='avg',size=(2,2),stride=(2,2),padding='SAME')
        return  avg_pool_

    def dense_block(self,input_x,name,growth_tate=12,L=5):
        '''
        DenseNet基本模块，每个bottleneck_layer直接是dense connect
        输入[N,H,W,K0],输出[N,H,W,K0+K*L]
        :param input_x: 输入
        :param bn_istraning:BN训练标志位
        :param name:
        :param growth_tate: 每个bottleneck_layer的输出是[N,H,W,K]
        :param L: bottleneck_layer的层数
        :return:
        '''
        con_stat=input_x
        bn_istraining=self.bn_istraining
        with tf.variable_scope(name):
            for node_ in range(L):
                node_name='dense_'+str(node_+1)
                #每次输出为K=12
                layer_op=self._bottleneck_layer(input_x=con_stat,growth_rate=growth_tate,bn_istraning=bn_istraining,name=node_name)
                #con_stat叠加L次，最后的输出为 K0+K*L
                con_stat=tf.concat([con_stat,layer_op],axis=3,name='concat'+str(node_+1))
        return con_stat

    def _bottleneck_layer(self,input_x,growth_rate,bn_istraning,name):
        '''
        最小单位：1X1 CONV -> BN -> RELU -> 3X3 CONV -> BN -> RELU

        :param input_x: 输入
        :param growth_rate: 每个点输出的feature map 数量 K，一般比较小，如12
        :param bn_istraning: BN是否在训练的标志位
        :param name: 每层名字
        :return:[N,H,W,K0]->[N,H,W,4*K]->[N,H,W,2*K]
        '''
        input_depth=int(input_x.get_shape()[-1])
        #1X1 conv  (N,H,W,C0)->(N,H,W,4K)
        bottle_=self._convblock(x=input_x,name=name+r'/bottleneck_1X1',shape=[1,1,input_depth,4*growth_rate],
                                bn_istraing=bn_istraning,padding='SAME',stride=[1,1])
        #3X3 CONV (N,H,W,4K)->(N,H,W,K)
        composite_=self._convblock(x=bottle_,name=name+r'/composite_3X3',shape=[3,3,4*growth_rate,growth_rate],
                                   bn_istraing=bn_istraning,padding='SAME',stride=[1,1])
        return composite_
    def fc_layer(self,x,name,shape):
        '''
        全连接层函数
        :param x: 输入
        :param name: 层名称
        :return:  X*W+B 没有relu
        '''
        with tf.variable_scope(name):
            w=tf.get_variable('weight',shape=shape)
            b=tf.get_variable('biases',shape=shape[-1:])
            return tf.matmul(x,w)+b
