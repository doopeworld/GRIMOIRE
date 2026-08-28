# =====================================================================
#  b70 -- Intel Arc Pro B70 (Battlemage G31) inference engine
#
#  Package names verified against Intel dgpu-docs + Ubuntu mirrors.
#  The old names (intel-level-zero-gpu, level-zero) pull the dead
#  libigc1 generation and cannot resolve. Current names:
#      intel-level-zero-gpu -> libze-intel-gpu1
#      level-zero           -> libze1
#      ocloc                -> intel-ocloc   (own package, AOT needs it)
#  Repo is the kobuk-team PPA, not repositories.intel.com/gpu.
# =====================================================================
FROM ubuntu:24.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
ARG B70_TARGET=intel_gpu_bmg_g31

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gpg gpg-agent wget make software-properties-common \
 && rm -rf /var/lib/apt/lists/*

# GPU runtime + ocloc
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 intel-ocloc intel-opencl-icd \
 && rm -rf /var/lib/apt/lists/*

# oneAPI DPC++ compiler
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends intel-oneapi-compiler-dpcpp-cpp \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Host tests: no GPU needed, catch regressions before the slow SYCL pass
RUN make test

# SYCL build. Never aborts the image: collect ALL errors in one pass
# instead of one per rebuild.
RUN ( B70_TARGET=${B70_TARGET} bash ./build_b70.sh 2>&1 | tee /src/build.log ) || true; \
    if [ -s /src/b70_native_inference ]; then echo "=== SYCL BUILD OK ==="; \
    else echo "=== SYCL BUILD FAILED ==="; grep -iE "error" /src/build.log | head -40; \
         printf '#!/bin/sh\necho "=== SYCL BUILD FAILED - errors below ==="\ngrep -iE "error|Error" /src/build.log | head -40\n' > /src/b70_native_inference; \
         chmod +x /src/b70_native_inference; fi

# =====================================================================
#  Single stage on purpose.
#
#  A slim runtime stage saves ~6 GB but drops the SYCL Unified Runtime
#  adapter plugins (libur_adapter_level_zero / _opencl). Without them
#  SYCL enumerates ZERO devices even though Level Zero and OpenCL both
#  see the cards perfectly -- which is exactly what happened here.
#  Correctness beats image size.
# =====================================================================

ENV ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# The Level Zero V2 adapter mis-handles allocation on Battlemage: a small
# malloc_device after a large one faults. Documented against dual Arc Pro
# B70 with this exact model architecture. V1 works.
ENV SYCL_UR_USE_LEVEL_ZERO_V2=0
ENV UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
ENV SYCL_CACHE_PERSISTENT=1
ENV SYCL_CACHE_DIR=/cache/sycl
VOLUME ["/models", "/cache"]
WORKDIR /src

# setvars must be sourced for the UR adapters to be on the library path.
RUN printf '#!/bin/bash\nset +u\nsource /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true\nexec "$@"\n' \
      > /usr/local/bin/entry.sh && chmod +x /usr/local/bin/entry.sh \
 && cp /src/b70_native_inference /usr/local/bin/ 2>/dev/null || true \
 && cp /src/bin/* /usr/local/bin/ 2>/dev/null || true \
 && chmod +x /usr/local/bin/b70-inspect /usr/local/bin/b70-load /usr/local/bin/grimoire /usr/local/bin/b70-verify-tok 2>/dev/null || true \
 && cp /src/tools/preflight.sh /usr/local/bin/ 2>/dev/null || true \
 && cp /src/tools/sweep.sh /usr/local/bin/ 2>/dev/null || true \
 && chmod +x /usr/local/bin/sweep.sh /usr/local/bin/preflight.sh 2>/dev/null || true

# ld.so config rather than LD_LIBRARY_PATH: --entrypoint bypasses the
# wrapper script, and then nothing finds libsycl. Baking the paths in
# makes every binary in the image runnable directly.
RUN find /opt/intel/oneapi -name 'libsycl.so*' -printf '%h\n' | sort -u > /etc/ld.so.conf.d/oneapi.conf \
 && find /opt/intel/oneapi -name 'libur_adapter*' -printf '%h\n' | sort -u >> /etc/ld.so.conf.d/oneapi.conf \
 && find /opt/intel/oneapi -name 'libtbb.so*' -printf '%h\n' | sort -u >> /etc/ld.so.conf.d/oneapi.conf \
 && ldconfig \
 && ldd /usr/local/bin/grimoire | grep -q "not found" && echo "WARNING: unresolved libs" || echo "all libs resolved"

ENTRYPOINT ["/usr/local/bin/entry.sh"]
CMD ["/usr/local/bin/b70_native_inference"]
