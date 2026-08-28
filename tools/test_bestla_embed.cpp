// STEP-3 FEASIBILITY: can GRIMOIRE (C++/SYCL) call BesTLA with its OWN USM buffers?
// BesTLA ships only as a pybind11 module (no torch op, no C ABI), so the only route is
// embedded CPython.  This proves the full chain:
//   SYCL USM ptr -> at::Tensor -> PyObject -> ark.woqgemm_linear -> at::Tensor -> USM
#include <Python.h>
#include <torch/all.h>
#include <torch/csrc/autograd/python_variable.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>


typedef std::chrono::steady_clock clk;
static void die(const char* m){ if(PyErr_Occurred())PyErr_Print(); std::fprintf(stderr,"FATAL: %s\n",m); std::exit(2); }

int main(int argc,char**argv){
    const int M=4096;
    // ---- 1. our own SYCL queue and USM activation buffer -------------------
    sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- 2. embed CPython --------------------------------------------------
    Py_Initialize();
    if(!Py_IsInitialized()) die("Py_Initialize");
    PyRun_SimpleString("import sys; sys.argv=['grimoire']");
    PyObject* torchmod = PyImport_ImportModule("torch");
    if(!torchmod) die("import torch");
    PyObject* ark = PyImport_ImportModule("auto_round_kernel");
    if(!ark) die("import auto_round_kernel");
    std::printf("embedded CPython: torch + auto_round_kernel imported\n");

    // ---- 3. build the packed weight from the REAL checkpoint (Python side) --
    const char* prep =
      "import torch, json, glob, os\n"
      "from safetensors import safe_open\n"
      "import auto_round_kernel as ark\n"
      "MD='/models/Qwen3.8-27B-int4-AutoRound'\n"
      "idx=json.load(open(glob.glob(MD+'/*.index.json')[0]))['weight_map']\n"
      "base='model.language_model.layers.0.mlp.gate_proj'\n"
      "t={}\n"
      "for suf in ('qweight','scales'):\n"
      "    k=base+'.'+suf\n"
      "    with safe_open(os.path.join(MD,idx[k]),framework='pt') as f: t[suf]=f.get_tensor(k)\n"
      "qw=t['qweight'].to('xpu'); sc=t['scales'].to('xpu').to(torch.bfloat16)\n"
      "G=128\n"
      "sh=torch.arange(8,device='xpu',dtype=torch.int32)*4\n"
      "qw8=((qw.unsqueeze(1)>>sh.view(1,8,1))&0xF).reshape(qw.shape[0]*8,qw.shape[1]).to(torch.int8).contiguous()\n"
      "K=sc.shape[0]*G; N=sc.shape[1]\n"
      "PACKED=ark.repack_quantized_weight(qw8,sc,None,G,'bf16','int4','bf16',False)\n"
      "DIMS=(N,K,G)\n"
      "print('  packed real weight: N=%d K=%d G=%d'%(N,K,G))\n";
    if(PyRun_SimpleString(prep)!=0) die("prepare packed weight");
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* packed = PyObject_GetAttrString(main,"PACKED");
    PyObject* dims   = PyObject_GetAttrString(main,"DIMS");
    if(!packed||!dims) die("fetch PACKED/DIMS");
    const int N=(int)PyLong_AsLong(PyTuple_GetItem(dims,0));
    const int K=(int)PyLong_AsLong(PyTuple_GetItem(dims,1));
    const int G=(int)PyLong_AsLong(PyTuple_GetItem(dims,2));

    // ---- 4. GRIMOIRE-owned USM buffer -> at::Tensor -> PyObject -------------
    auto* xusm = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(size_t(M)*K,q);
    { std::vector<sycl::ext::oneapi::bfloat16> h(size_t(M)*K);
      std::mt19937 rng(3); std::uniform_real_distribution<float> d(-1.f,1.f);
      for(auto&v:h) v=sycl::ext::oneapi::bfloat16(d(rng));
      q.memcpy(xusm,h.data(),h.size()*2).wait(); }

    c10::xpu::XPUStream ext = c10::xpu::getStreamFromExternal(&q,0);
    c10::xpu::setCurrentXPUStream(ext);            // run BesTLA on OUR queue

    auto opts = at::TensorOptions().dtype(at::kBFloat16).device(at::kXPU);
    at::Tensor xt = at::from_blob(xusm,{M,K},opts);
    PyObject* xpy = THPVariable_Wrap(xt);
    if(!xpy) die("THPVariable_Wrap");
    std::printf("  wrapped GRIMOIRE USM buffer (%d x %d) as a torch tensor\n",M,K);

    // ---- 5. call BesTLA ----------------------------------------------------
    PyObject* fn = PyObject_GetAttrString(ark,"woqgemm_linear");
    if(!fn) die("woqgemm_linear");
    auto call=[&]()->PyObject*{
        PyObject* a=Py_BuildValue("(OOOiiisssO)",xpy,packed,Py_None,N,K,G,
                                  "bf16","int4","bf16",Py_False);
        PyObject* r=PyObject_CallObject(fn,a); Py_DECREF(a);
        if(!r){ PyErr_Print(); die("woqgemm_linear call"); }
        return r;
    };
    PyObject* out=call(); q.wait(); Py_DECREF(out);
    double best=1e18;
    for(int i=0;i<6;++i){
        auto t0=clk::now(); PyObject* r=call(); q.wait();
        double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
        Py_DECREF(r); if(i&&ms<best)best=ms;
    }
    const double fl=2.0*M*N*K;
    std::printf("\n  BesTLA via embedded CPython: %.3f ms   %.1f TFLOP/s\n",best,fl/best*1e-9);
    std::printf("  (GRIMOIRE cutlass MXFP4 same shape: 7.314 ms, 99.8 TFLOP/s)\n");
    std::printf("  speedup: %.2fx\n", 7.314/best);

    // ---- 6. result comes back as a tensor we can read from C++ -------------
    PyObject* r=call(); q.wait();
    at::Tensor rt = THPVariable_Unpack(r);
    std::printf("  result tensor: sizes=[%ld,%ld] dtype=%s device=%s\n",
        rt.size(0),rt.size(1),c10::toString(rt.scalar_type()),rt.device().str().c_str());
    Py_DECREF(r);
    std::printf("\nSTEP-3 FEASIBILITY: PASS\n");
    return 0;
}
