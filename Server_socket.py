from socket import *

address='172.27.34.3'    
port=1234           #监听自己的哪个端口
buffsize=1024          #接收从客户端发来的数据的缓存区大小
print('\n主机ip{},开放端口:{}\n等待连接.......'.format(address,port))
s = socket(AF_INET, SOCK_STREAM)
s.bind((address,port))
s.listen(1)     #最大连接数
clientsock,clientaddress=s.accept()
print('connect from:',clientaddress)
#传输数据都利用clientsock，和s无关
while True: 
    #print("receive")
    recvdata=clientsock.recv(buffsize).decode('utf-8')
    print(recvdata)
    if recvdata=='exit' or not recvdata:
        print("exit program!")
        break
    senddata=recvdata+'from sever'
    #clientsock.send(senddata.encode())
clientsock.close()
s.close()
#添加可用的端口
#sudo iptables -I INPUT -p tcp --dport 1234 -j ACCEPT
#sudo iptables-save
