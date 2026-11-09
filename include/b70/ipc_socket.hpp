// SPDX-License-Identifier: GPL-3.0-or-later
// Small, CPU-only control messages and Linux descriptor transfer for GPU IPC.
#pragma once
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace b70::ipc {
inline bool wait_ready(int fd, short events) {
    pollfd p{fd,events,0}; int n;
    do { n=::poll(&p,1,120000); } while(n<0 && errno==EINTR);
    return n>0 && (p.revents&events);
}
inline bool write_all(int fd, const void* data, size_t bytes) {
    auto p = static_cast<const char*>(data);
    while (bytes) {
        if(!wait_ready(fd,POLLOUT)) return false;
        const ssize_t n = ::send(fd, p, bytes, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n; bytes -= size_t(n);
    }
    return true;
}
inline bool read_all(int fd, void* data, size_t bytes) {
    auto p = static_cast<char*>(data);
    while (bytes) {
        if(!wait_ready(fd,POLLIN)) return false;
        const ssize_t n = ::recv(fd, p, bytes, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n; bytes -= size_t(n);
    }
    return true;
}
template<class T> bool exchange(int fd, bool first, const T& own, T& peer) {
    return first ? write_all(fd, &own, sizeof own) && read_all(fd, &peer, sizeof peer)
                 : read_all(fd, &peer, sizeof peer) && write_all(fd, &own, sizeof own);
}
// Exactly one byte carries SCM_RIGHTS; the opaque handle follows as ordinary
// bytes. Descriptor numbers are process-local and must never be sent as ints.
inline bool send_fd(int socket, int descriptor) {
    char byte = 'F'; iovec io{&byte, 1};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
    msghdr msg{}; msg.msg_iov = &io; msg.msg_iovlen = 1;
    msg.msg_control = control; msg.msg_controllen = sizeof control;
    auto c = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(c), &descriptor, sizeof descriptor);
    ssize_t n;
    if(!wait_ready(socket,POLLOUT)) return false;
    do { n = ::sendmsg(socket, &msg, MSG_NOSIGNAL); } while (n < 0 && errno == EINTR);
    return n == 1;
}
inline bool recv_fd(int socket, int& descriptor) {
    descriptor = -1;
    char byte = 0; iovec io{&byte, 1};
    alignas(cmsghdr) char control[CMSG_SPACE(8 * sizeof(int))]{};
    msghdr msg{}; msg.msg_iov = &io; msg.msg_iovlen = 1;
    msg.msg_control = control; msg.msg_controllen = sizeof control;
    ssize_t n;
    if(!wait_ready(socket,POLLIN)) return false;
    do { n = ::recvmsg(socket, &msg, MSG_CMSG_CLOEXEC); } while (n < 0 && errno == EINTR);
    int count = 0;
    for (auto c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c)) {
        if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS || c->cmsg_len < CMSG_LEN(0)) continue;
        const size_t nf = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        for (size_t i = 0; i < nf; ++i) {
            int fd; std::memcpy(&fd, CMSG_DATA(c) + i * sizeof(int), sizeof fd);
            if (++count == 1) descriptor = fd; else ::close(fd);
        }
    }
    if (n != 1 || byte != 'F' || count != 1 || (msg.msg_flags & (MSG_CTRUNC | MSG_TRUNC))) {
        if (descriptor >= 0) ::close(descriptor);
        descriptor = -1; return false;
    }
    return true;
}
} // namespace b70::ipc
