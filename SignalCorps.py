'''
2018年5月17日10:54:33
    通信脚本
    主要实现功能，从iTracker中接受信息
    再通过 TCP/IP 发送
'''
from socket import *
import time
class signalcorps():
    def __init__(self,HOST='172.27.34.1',PORT=1234):
        #InfoPOOL 作为信息池，存储需要发送的信息
        self.InfoPOOL=[]
        self.sendCounter=0
        try:
            clientsocket = socket(AF_INET,SOCK_STREAM)
            clientsocket.connect((HOST,PORT))
            self.clientsocket=clientsocket
        except :
            print("初次连接失败！")
            self.reconnet((HOST,PORT))

    def reconnet(self,host_prot,waittime=5,loop=3):
        print("尝试重新连接！")
        for i in range(loop):
            print("第 {} 次尝试:休眠 {} 秒后重新连接".format(i+1,waittime))
            time.sleep(waittime)
            try:
                clientsocket = socket(AF_INET,SOCK_STREAM)
                clientsocket.connect(host_prot)
                self.clientsocket=clientsocket
                break
            except :
                self.clientsocket=None

        if self.clientsocket is None:
            raise Exception("TCP/IP连接失败，无法建立通信！")
        else:
            print("连接成功！")
    def writeinfo(self,info,infotype='normal'):
        if infotype=="pick":
            write_info=",".join([str(i_) for i_ in info])
            info=write_info
        elif infotype=="command":
            #UserCommand 中输入的信息为 0~8 的某个数
            # print(info)
            # info_=[0]*8
            # info_[info]=1
            # info_=''.join([str(i) for i in info_[::-1]])
            # info_=int(info_,2)
            # print(info_)
            info=str(info)#1,3,4,5,7
            #1:0001
            #3:0011
            #4:0100
            #5:0101
            #7:0111
        self.InfoPOOL.append(info)

    def sendAll(self):
        #发送所有信息给从机
        #暂时用打印代替发送
        clientsocket=self.clientsocket
        self.sendCounter+=1
        for info_ in self.InfoPOOL:
            #print("???")

            data=info_.encode()
            clientsocket.send(data)#发送数据
            #recv_data = clientsocket.recv(1024)#回传数据，可能不需要
        self.InfoPOOL=[]#清空信息池子

    def cleanInfoPool(self):
        self.InfoPOOL=[]

class signalcorps_localtest():
    ''''
    测试使用，用来而是cnn网络的准确度，调用这个类的话
    实际上并没有连接服务器
    '''
    def __init__(self,HOST='172.27.34.1',PORT=1234):
        print("LOCAL TEST!!!!")
    def reconnet(self,host_prot,waittime=5,loop=3):
        print("LOCAL TEST!!!!")
    def writeinfo(self,info,infotype='normal'):
        print("LOCAL TEST!!!!")
    def sendAll(self):
        print("LOCAL TEST!!!!")
    def cleanInfoPool(self):
        print("LOCAL TEST!!!!")
if __name__ == '__main__':
    #测试通信
    print("通讯模块测试：与Server链接测试！")
    sc=signalcorps(PORT=8848)
    sc.writeinfo("HelloWord")
    sc.sendAll()
    for i in range(3):
        info_data=input("type\n")
        sc.writeinfo(info_data)
        sc.sendAll()
