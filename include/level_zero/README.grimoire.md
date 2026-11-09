`ze_api.h` is the unmodified MIT-licensed Intel Level Zero API header from
https://github.com/oneapi-src/level-zero/blob/v1.24.2/include/ze_api.h

Pinned because the existing oneAPI compiler image supplies SYCL's native
interop declarations but does not install the Level Zero development package.
P2P dynamically loads the installed runtime; this header adds no runtime library.
