// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/ipc_socket.hpp"
#include <sys/wait.h>
#include <cstdio>
#include <vector>

int main() {
    int s[2],p[2]; if(socketpair(AF_UNIX,SOCK_STREAM,0,s)||pipe(p))return 1;
    // Verify the received descriptor is usable and is closed on exec.
    if(!b70::ipc::send_fd(s[0],p[0]))return 2;
    int received=-1;if(!b70::ipc::recv_fd(s[1],received))return 3;
    char c='G',got=0;
    if(write(p[1],&c,1)!=1 || read(received,&got,1)!=1 || got!=c ||
       !(fcntl(received,F_GETFD)&FD_CLOEXEC))return 4;
    close(received);close(p[0]);close(p[1]);
    // Malformed control message must reject and close BOTH received handles.
    int nullfd=open("/dev/null",O_RDONLY),pass[2]={nullfd,nullfd};
    char byte='F'; iovec io{&byte,1}; alignas(cmsghdr) char control[CMSG_SPACE(sizeof pass)]{};
    msghdr msg{};msg.msg_iov=&io;msg.msg_iovlen=1;msg.msg_control=control;msg.msg_controllen=sizeof control;
    auto cm=CMSG_FIRSTHDR(&msg);cm->cmsg_level=SOL_SOCKET;cm->cmsg_type=SCM_RIGHTS;cm->cmsg_len=CMSG_LEN(sizeof pass);
    std::memcpy(CMSG_DATA(cm),pass,sizeof pass);
    if(sendmsg(s[0],&msg,0)!=1 || b70::ipc::recv_fd(s[1],received) || received!=-1)return 5;
    close(nullfd);
    // A payload larger than the socket buffer checks short I/O handling.
    pid_t kid=fork();if(kid<0)return 6;
    if(!kid) {
        close(s[0]);std::vector<unsigned char> data(4*1024*1024);
        bool ok=b70::ipc::read_all(s[1],data.data(),data.size());
        for(size_t i=0;ok && i<data.size();++i)ok=data[i]==i%251;
        close(s[1]);_exit(ok?0:7);
    }
    close(s[1]);std::vector<unsigned char> data(4*1024*1024);
    for(size_t i=0;i<data.size();++i)data[i]=i%251;
    if(!b70::ipc::write_all(s[0],data.data(),data.size()))return 8;
    close(s[0]);int status=0;
    if(waitpid(kid,&status,0)<0||!WIFEXITED(status)||WEXITSTATUS(status))return 9;
    puts("IPC_SOCKET_PASS");return 0;
}
