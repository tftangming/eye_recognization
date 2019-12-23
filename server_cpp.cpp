
#include "stdafx.h"
#include <WINSOCK2.H>   
#include <stdio.h>     
#include "stdlib.h"
#define PORT           1234    
#define MSGSIZE        1024    
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#pragma comment(lib, "ws2_32.lib")    
int main()
{
	printf("等待连接....\n");
	WSADATA wsaData;
	SOCKET sListen;
	SOCKET sClient;
	SOCKADDR_IN local;
	SOCKADDR_IN client;
	char szMessage[MSGSIZE];
	int ret;
	int iaddrSize = sizeof(SOCKADDR_IN);
	WSAStartup(0x0202, &wsaData);

	sListen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	local.sin_family = AF_INET; 
	local.sin_port = htons(PORT);
	local.sin_addr.s_addr = htonl(INADDR_ANY);
	bind(sListen, (struct sockaddr *) &local, sizeof(SOCKADDR_IN));

	listen(sListen, 1);
	
	sClient = accept(sListen, (struct sockaddr *) &client, &iaddrSize);
	printf("Accepted client:%s:%d\n", inet_ntoa(client.sin_addr),
		ntohs(client.sin_port));
	int hh=1;
	int hc = 1;
	//int atoi(const char *nptr);
	while (TRUE) {
		ret = recv(sClient, szMessage, MSGSIZE, 0);
		szMessage[ret] = '\0'; 
		printf("Received [%d bytes]: '%s'\n", ret, szMessage);
		hc = atoi(szMessage);
		//sscanf(szMessage, "%d", &hh);
		printf("%d\n", hc);
		
		if (ret ==0)
		{
			break;
		}
	}
	return 0;
}