# Compiling GRIMOIRE without the Tower

Several sessions of this project shipped code that had never been through
a compiler, on the assumption that a work container has no SYCL toolchain
and nothing can be done about it. That assumption is wrong, and it cost
real bugs: the first `icpx -fsyntax-only` run against `src/grimoire.cpp`
on 2026-09-12 failed on a duplicate variable declaration, and the first
execution of the new K2 kernels found a precision defect in the softplus
gate and four engine bugs in the K2 path.

**Set this up before writing engine code, not after.** It takes about ten
minutes and needs no GPU.

## What you get

- `icpx` — the real Intel oneAPI DPC++ compiler, so every SYCL source
  compiles for `spir64` with full device-code generation.
- An OpenCL **CPU** device, so kernels actually RUN and can be diffed
  against the host references.

## What you do NOT get

- No XMX/DPAS. Anything using `joint_matrix` — `gemm_xmx`, the whole
  batched prefill path — cannot execute. `Grimoire::prefill` detects the
  missing `ext_intel_matrix` aspect and falls back to sequential.
- No AOT image for `intel_gpu_bmg_g31` (that needs `ocloc`, and see the
  note below -- an `ocloc` that INSTALLS is not necessarily one that can
  target Battlemage), no
  bandwidth, no timing, no B70 driver behaviour.
- Therefore: **no benchmark number from here means anything.** Rule 8 is
  unchanged. This is a correctness floor, not a substitute for the card.

## Install

Packages come from conda-forge over plain HTTPS; no conda needed, they
are zip files containing zstd tarballs.

```bash
pip install zstandard
cd "$SCRATCH"
BASE=https://conda.anaconda.org/conda-forge/linux-64
for f in dpcpp_impl_linux-64-2026.1.1-h2a08c78_328.conda \
         dpcpp_linux-64-2026.1.1-hf83752f_328.conda \
         intel-cmplr-lib-rt-2026.1.1-h04fd95a_328.conda \
         intel-cmplr-lib-ur-2026.1.1-h04fd95a_328.conda \
         intel-cmplr-lic-rt-2026.1.1-ha770c72_328.conda \
         intel-sycl-rt-2026.1.1-h7a4b287_328.conda \
         intel-opencl-rt-2026.1.1-he68e48a_328.conda \
         ocl-icd-2.3.4-hb03c661_1.conda \
         umf-1.1.0-h7d1de2b_343.conda \
         tbb-2022.3.0-hb700be7_2.conda ; do
  curl -sS -o "pkgs/$f" "$BASE/$f"
done

python3 - <<'PY'
import zipfile, zstandard, tarfile, io, glob, os
d = zstandard.ZstdDecompressor()
for p in sorted(glob.glob('pkgs/*.conda')):
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if not n.endswith('.tar.zst') or n.startswith('info-'): continue
            buf = io.BytesIO(d.decompress(z.read(n), max_output_size=4*1024**3))
            with tarfile.open(fileobj=buf) as t: t.extractall('/opt/dpcpp', filter='tar')
    print('extracted', os.path.basename(p))
PY

apt-get install -y opencl-headers libhwloc15      # CL/cl.h, libhwloc.so.15
echo /opt/dpcpp/lib/intel-ocl-cpu/libintelocl.so \
   > /opt/dpcpp/etc/OpenCL/vendors/intel-ocl-cpu.icd   # the shipped .icd
                                                       # points at a conda
                                                       # build placeholder
```

Environment:

```bash
export PATH=/opt/dpcpp/bin:$PATH
export LD_LIBRARY_PATH=/opt/dpcpp/lib:/opt/dpcpp/lib/x86_64-unknown-linux-gnu:/opt/dpcpp/lib/intel-ocl-cpu:$LD_LIBRARY_PATH
export OCL_ICD_VENDORS=/opt/dpcpp/etc/OpenCL/vendors
sycl-ls     # expect: [opencl:cpu][opencl:0] Intel(R) OpenCL, ...
```

If `sycl-ls` says "No platforms found", run `sycl-ls --verbose`: it names
the missing `.so` directly. The three that bite are `CL/cl.h` at compile
time, then `libOpenCL.so.1` (ocl-icd), `libumf.so.1` (umf) and
`libhwloc.so.15` at load time.

## Use

Compile with `-fsycl-targets=spir64` instead of `intel_gpu_bmg_g31` — the
AOT target needs `ocloc`, and SPIR-V still runs the full front end and
device code generation:

```bash
# fastest signal on one file
icpx -fsycl -fsycl-targets=spir64 -fsyntax-only -std=c++20 -ferror-limit=0 \
     -I include -I src src/grimoire.cpp

# the real thing
icpx -fsycl -fsycl-targets=spir64 -O2 -std=c++20 -ferror-limit=0 \
     -fno-fast-math -ffp-contract=fast -fno-math-errno \
     -fsycl-device-code-split=off -I include -I src \
     tools/grimoire_main.cpp src/grimoire.cpp ... -o /tmp/grimoire
```

Then run the two device gates:

```bash
./bin/test_k2_kernels                      # kernel-vs-reference parity
GRIMOIRE_DEVICE_ANY=1 ./bin/test_k2_e2e    # the engine, end to end
```

**`-fsycl-device-code-split=off` puts every kernel in one image**, so the
CPU runtime JITs the `joint_matrix` kernels even when nothing calls them,
and dies. Use `-fsycl-device-code-split=per_kernel` for anything you
intend to RUN here. `off` is correct for the B70 and is what
`build_b70.sh` uses — keep it there.

## The rule this exists to enforce

A change to `src/*.cpp` is not finished until `icpx` has accepted it and,
where a test exists, the kernels have run. "Written but never compiled"
is not a state to hand off in.

## `ocloc`: present is not the same as usable (MEASURED 2026-09-13)

`apt-get install intel-ocloc` succeeds on Ubuntu 24.04 and puts a working
`ocloc` on PATH. It is version 23.43, which predates Battlemage: its
device list stops at `pvc` / `mtl-*` / `xe-lpg`, and

```
$ ocloc compile -device bmg_g31 -file x.cl
Could not determine device target: bmg_g31.
Error: Cannot get HW Info for device bmg_g31.
```

So "install ocloc and build the AOT image" does not work from the distro
package. Battlemage needs Intel's own compute-runtime, 24.35 or newer,
and `repositories.intel.com` was NOT reachable from this work container
(403 through the egress proxy). The AOT image is therefore Tower work,
for a specific reason rather than a vague one.

`build_b70.sh` now probes each die with `ocloc ids` before it compiles
anything and refuses with that explanation, instead of letting the build
run to the device compile and fail with an error that names the die. Note
that `ocloc ids` exits 0 whether or not it recognises the acronym, so the
check reads its OUTPUT; an `ocloc` too old to have the subcommand at all
is not treated as a refusal.
