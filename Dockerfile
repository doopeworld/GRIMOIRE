# =====================================================================
# GRIMOIRE -- bare-metal SYCL/Level-Zero/C++ inference engine for the
# Intel Arc Pro B70 (Battlemage G31); the same build runs on the Arc Pro B60
# and Arc B580 (Battlemage G21).
#
# NO vLLM. NO PyTorch. NO Python ML stack, anywhere in this image, at any
# stage. Every compute kernel is this project's own SYCL/ESIMD source,
# except the prompt GEMMs that go through Intel oneDNN (Apache-2.0, built
# from source below with the same compiler); the other third-party pieces
# are Intel's own build toolchain (oneAPI DPC++) and the Intel GPU driver
# userspace stack (Level Zero / OpenCL / IGC), all pinned to exact versions.
#
# Build:  docker build -t grimoire-b70 .
# Run:    see README.md "Docker" section for the full, safe invocation
#         (--init and a generous --stop-timeout are REQUIRED -- see
#         docker-entrypoint.sh and the comment on BUILDER_STOP_TIMEOUT
#         below for why).
# =====================================================================

# ---------------------------------------------------------------------
# Stage 1: builder.  Compiles Intel oneDNN, then bin/grimoire,
# bin/grimoire-server, bin/libgrimoire_gemm.so and bin/libgrimoire_attn.so
# with Intel's oneAPI DPC++ compiler.  Nothing from this stage reaches the
# final image except those files, libdnnl.so and oneDNN's license files.
# ---------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

# Pinned to the exact compiler build the kernels were checked against: icpx 2026.2.0 with
# compute-runtime 26.35 / IGC 2.41.5 and oneDNN 3.14 since v1.10.0 (2026-10-28: against
# icpx 2026.1.1 / 26.27 / IGC 2.38.2 / oneDNN 3.13.4, tools/regress_all_g0.sh gave 43 of 44
# answers identical, the 44th a one-word rounding flip, at the same speed on one and two
# B70s).  A different version is not guaranteed to generate the same ESIMD/DPAS code; if you
# need a newer oneAPI, re-run tools/regress_all_g0.sh against it before trusting the
# numbers in any commit message.
ARG ONEAPI_VERSION=2026.2
ARG COMPUTE_RUNTIME_VERSION=26.35.39758.10
ARG IGC_VERSION=2.41.5
ARG IGC_BUILD=22716
ARG GMMLIB_VERSION=22.10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget build-essential ocl-icd-libopencl1 \
    && wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "intel-oneapi-compiler-dpcpp-cpp-${ONEAPI_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# AOT codegen for intel_gpu_bmg_g31 needs `ocloc`, which is NOT part of the
# compiler package above -- it ships in Intel's GPU compute-runtime release,
# the same one the final image's GPU driver comes from. Without this,
# icpx's AOT step fails with "ocloc tool could not be found".
RUN mkdir -p /tmp/neo && cd /tmp/neo \
    && wget -q "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
    && wget -q "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
    && wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
    && wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMMLIB_VERSION}_amd64.deb" \
    && dpkg -i *.deb \
    && cd / && rm -rf /tmp/neo

# Intel oneDNN for the prompt GEMMs (src/onednn_gemm.cpp, on by default since
# 2026-10-25; GRIMOIRE_ONEDNN=0 turns it off): SYCL GPU runtime only -- no CPU
# primitives, no graph API, no tests -- the configuration measured in
# tools/bench/onednn/RESULTS-2026-10-25-ONEDNN.md.  The build scripts below find
# it in third_party/onednn and put libdnnl.so next to the binaries (rpath $ORIGIN).
ARG ONEDNN_VERSION=v3.14
# parallel compile jobs for oneDNN (the build host keeps the rest of its cores)
ARG BUILD_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch ${ONEDNN_VERSION} https://github.com/uxlfoundation/oneDNN.git /tmp/onednn \
    && bash -c 'set -e; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; \
        CC=icx CXX=icpx cmake -S /tmp/onednn -B /tmp/onednn/build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/grimoire/third_party/onednn \
          -DONEDNN_CPU_RUNTIME=NONE -DONEDNN_GPU_RUNTIME=SYCL -DONEDNN_GPU_VENDOR=INTEL \
          -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_GRAPH=OFF; \
        cmake --build /tmp/onednn/build --target dnnl -j ${BUILD_JOBS}; \
        cmake --install /tmp/onednn/build >/dev/null' \
    && mkdir -p /grimoire/licenses/onednn \
    && cp /tmp/onednn/LICENSE /tmp/onednn/THIRD-PARTY-PROGRAMS /grimoire/licenses/onednn/ \
    && rm -rf /tmp/onednn

WORKDIR /grimoire
COPY src/        src/
COPY include/    include/
COPY tools/build_grimoire_only_b70.sh  tools/
COPY tools/build_server_only_b70.sh    tools/
COPY tools/grimoire_main.cpp           tools/
COPY tools/grimoire_server.cpp         tools/
RUN mkdir -p bin

# GPU targets for the AOT kernels (tools/build_*_only_b70.sh read it): the Arc Pro B70
# (BMG G31) and the Arc Pro B60 / Arc B580 (BMG G21), so the same image runs on both;
# --build-arg GRIM_TARGETS=intel_gpu_bmg_g31 builds the B70 only, at half the
# device-compile time.
ARG GRIM_TARGETS=intel_gpu_bmg_g31,intel_gpu_bmg_g21

# libdnnl.so.3 (the soname the binaries load) is copied out as a plain file, so the runtime
# stage below does not name a oneDNN version.
RUN bash tools/build_grimoire_only_b70.sh /grimoire \
    && bash tools/build_server_only_b70.sh /grimoire \
    && strip --strip-unneeded bin/grimoire bin/grimoire-server bin/libgrimoire_gemm.so bin/libgrimoire_attn.so \
    && cp -L bin/libdnnl.so.3 /grimoire/libdnnl.so.3

# ---------------------------------------------------------------------
# Stage 2: runtime.  Only what the three binaries actually need (checked
# with `ldd`, not assumed): the Level Zero / OpenCL GPU driver stack
# (exact versions, matching Battlemage support, pulled straight from
# Intel's own GitHub releases) and the oneAPI compiler's RUNTIME package
# (not the full compiler) for libsycl / libur_loader / the Intel math
# support libs. No -devel package, no Python, nothing else.
# ---------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ARG ONEAPI_VERSION=2026.2
ARG COMPUTE_RUNTIME_VERSION=26.35.39758.10
ARG IGC_VERSION=2.41.5
ARG IGC_BUILD=22716
ARG GMMLIB_VERSION=22.10.0
ARG LEVEL_ZERO_VERSION=1.32.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gnupg wget curl ocl-icd-libopencl1 \
    && wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "intel-oneapi-compiler-dpcpp-cpp-runtime-${ONEAPI_VERSION}" \
        "intel-oneapi-compiler-shared-runtime-${ONEAPI_VERSION}" \
        intel-oneapi-umf \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU compute-runtime / Level Zero: pulled as standalone .deb
# releases, the same way as every B70 driver install this project has
# measured against (apt.repos.intel.com's GPU feed lags new hardware
# support; these GitHub releases do not). wget/gnupg are purged only
# after this step -- they are needed for both the apt layer above and
# the direct downloads here.
RUN mkdir -p /tmp/neo && cd /tmp/neo \
    && wget -q "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
    && wget -q "https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
    && wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
    && wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
    && wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMMLIB_VERSION}_amd64.deb" \
    && wget -q "https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb" \
    && wget -q "https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/libze1_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb" \
    && dpkg -i *.deb \
    && cd / && rm -rf /tmp/neo \
    && apt-get purge -y wget gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Every oneAPI component installs its own lib/ under a DIFFERENT
# versioned subdirectory (compiler/2026.1/lib, umf/1.1/lib, ...) -- none
# of them are on the default linker search path. Discovering all of them
# (rather than hardcoding the compiler's alone) caught a real miss: the
# Level Zero adapter links libumf.so.1, which lives under umf/1.1/lib,
# not compiler/2026.1/lib -- without it the adapter dlopens silently
# fail and Level Zero reports zero GPU adapters with no error.
# ldconfig once at BUILD time (not LD_LIBRARY_PATH at run time) so this
# survives however the container is invoked.
RUN find /opt/intel/oneapi -mindepth 1 -maxdepth 3 -type d -name lib \
        > /etc/ld.so.conf.d/intel-oneapi.conf \
    && ldconfig

WORKDIR /grimoire
COPY --from=builder /grimoire/bin/grimoire           bin/grimoire
COPY --from=builder /grimoire/bin/grimoire-server     bin/grimoire-server
COPY --from=builder /grimoire/bin/libgrimoire_gemm.so  bin/libgrimoire_gemm.so
COPY --from=builder /grimoire/bin/libgrimoire_attn.so  bin/libgrimoire_attn.so
COPY --from=builder /grimoire/libdnnl.so.3 bin/libdnnl.so.3
COPY --from=builder /grimoire/licenses/onednn/        /usr/share/doc/onednn/
RUN cd bin && ln -sf libdnnl.so.3 libdnnl.so
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh bin/grimoire bin/grimoire-server

ENV ONEAPI_DEVICE_SELECTOR=level_zero:gpu

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# Default: start the OpenAI-compatible server. Override the whole command
# to run bin/grimoire for a one-shot CLI generation instead -- see
# README.md.
CMD ["server"]
