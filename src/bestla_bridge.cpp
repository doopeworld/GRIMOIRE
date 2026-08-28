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
static PyObject *g_ark=nullptr,*g_fn=nullptr;
static PyObject *g_gate=nullptr,*g_down=nullptr;   // per-layer lists of packed blobs
static int g_layers=0,g_Ng=0,g_Kg=0,g_Nd=0,g_Kd=0,g_G=128;
// vLLM uses int8 compute on XPU (qlinear.py:210) -- W4A8.  Measured 148.5 vs 127.0
// TFLOP/s for bf16 compute, both cosine 1.0000 against a reference matmul.
static const char* g_cdt = [](){ const char* e=std::getenv("GRIMOIRE_BESTLA_CDT");
                                 return (e&&*e)?e:"int8"; }();

// Pack every layer's FFN weights from the int4-AutoRound checkpoint.
// Rationale: those weights were quantized ONCE from BF16.  Re-quantizing GRIMOIRE's
// MXFP4 artifact into int4 g128 would be a second lossy step (per-32 E8M0 -> per-128
// bf16) and would degrade output.  The MXFP4 artifact stays in use for DECODE; this
// int4 copy is used only for PREFILL, where BesTLA is 1.47x faster.
extern "C" int grimoire_bestla_init_all(const char* ckpt,int nlayers,
                                        int* Ng,int* Kg,int* Nd,int* Kd){
    if(!Py_IsInitialized()){ Py_Initialize(); PyRun_SimpleString("import sys; sys.argv=['grimoire']"); }
    if(!Py_IsInitialized()) return -1;
    if(!PyImport_ImportModule("torch")){ PyErr_Print(); return -2; }
    g_ark=PyImport_ImportModule("auto_round_kernel");
    if(!g_ark){ PyErr_Print(); return -3; }
    PyObject* main=PyImport_AddModule("__main__");
    PyObject* d=PyModule_GetDict(main);
    PyDict_SetItemString(d,"CKPT",PyUnicode_FromString(ckpt));
    PyDict_SetItemString(d,"NL",PyLong_FromLong(nlayers));
    const char* prep =
      "import torch, json, glob, os, time\n"
      "from safetensors import safe_open\n"
      "import auto_round_kernel as ark\n"
      "G=128\n"
      "idx=json.load(open(glob.glob(CKPT+'/*.index.json')[0]))['weight_map']\n"
      "_open={}\n"
      "def gt(k):\n"
      "    f=idx[k]\n"
      "    if f not in _open: _open[f]=safe_open(os.path.join(CKPT,f),framework='pt')\n"
      "    return _open[f].get_tensor(k)\n"
      "CDT=os.environ.get('GRIMOIRE_BESTLA_CDT','int8')\n"
      "SH=torch.arange(8,dtype=torch.int32)*4\n"
      "ZP=int(os.environ.get('GRIMOIRE_BESTLA_ZP','8'))\n"
      "def unpack(qw):\n"
      "    # GPTQ packs 8 nibbles per int32 along K.  AutoRound here is SYMMETRIC\n"
      "    # (sym=True), so the stored nibbles are offset by 8; BesTLA's 'int4' with\n"
      "    # asym=False wants signed -8..7, hence the subtraction.\n"
      "    u=((qw.unsqueeze(1)>>SH.view(1,8,1))&0xF).reshape(qw.shape[0]*8,qw.shape[1])\n"
      "    return (u.to(torch.int16)-ZP).to(torch.int8).contiguous()\n"
      "def pack(base):\n"
      "    qw=unpack(gt(base+'.qweight')).to('xpu')\n"
      "    sc=gt(base+'.scales').to('xpu').to(torch.float16)\n"
      "    ZE=torch.empty(0,dtype=torch.int8,device='xpu')\n"
      "    return ark.repack_quantized_weight(qw,sc,ZE,G,CDT,'int4','fp16',False), sc.shape[1], sc.shape[0]*G\n"
      "GATE=[]; DOWN=[]\n"
      "t0=time.time()\n"
      "for L in range(NL):\n"
      "    b='model.language_model.layers.%d.mlp.'%L\n"
      "    pg,ng,kg=pack(b+'gate_proj'); pu,_,_=pack(b+'up_proj')\n"
      "    pd,nd,kd=pack(b+'down_proj')\n"
      "    GATE.append((pg,pu)); DOWN.append(pd)\n"
      "DIMS=(ng,kg,nd,kd)\n"
      "print('  BesTLA: packed %d layers in %.1fs'%(NL,time.time()-t0),flush=True)\n";
    if(PyRun_SimpleString(prep)!=0){ PyErr_Print(); return -4; }
    g_gate=PyObject_GetAttrString(main,"GATE");
    g_down=PyObject_GetAttrString(main,"DOWN");
    PyObject* dims=PyObject_GetAttrString(main,"DIMS");
    if(!g_gate||!g_down||!dims) return -5;
    g_Ng=(int)PyLong_AsLong(PyTuple_GetItem(dims,0));
    g_Kg=(int)PyLong_AsLong(PyTuple_GetItem(dims,1));
    g_Nd=(int)PyLong_AsLong(PyTuple_GetItem(dims,2));
    g_Kd=(int)PyLong_AsLong(PyTuple_GetItem(dims,3));
    g_layers=nlayers;
    g_fn=PyObject_GetAttrString(g_ark,"woqgemm_linear");
    if(!g_fn) return -6;
    *Ng=g_Ng;*Kg=g_Kg;*Nd=g_Nd;*Kd=g_Kd;
    return 0;
}

// which: 0 = gate_proj, 1 = up_proj, 2 = down_proj
extern "C" int grimoire_bestla_ffn(sycl::queue* q,int layer,int which,
                                   const void* x,void** outptr,int M){
    if(layer<0||layer>=g_layers) return -1;
    PyObject* packed=nullptr; int N,K;
    if(which<2){
        PyObject* pair=PyList_GetItem(g_gate,layer);
        packed=PyTuple_GetItem(pair,which); N=g_Ng; K=g_Kg;
    } else { packed=PyList_GetItem(g_down,layer); N=g_Nd; K=g_Kd; }
    if(!packed) return -2;
    c10::xpu::XPUStream ext=c10::xpu::getStreamFromExternal(q,0);
    c10::xpu::setCurrentXPUStream(ext);
    auto ob=at::TensorOptions().dtype(at::kBFloat16).device(at::kXPU);
    at::Tensor xt=at::from_blob(const_cast<void*>(x),{M,K},ob);
    PyObject* xpy=THPVariable_Wrap(xt);
    if(!xpy){ PyErr_Print(); return -3; }
    PyObject* a=Py_BuildValue("(OOOiiisssO)",xpy,packed,Py_None,N,K,g_G,
                              g_cdt,"int4","fp16",Py_False);
    PyObject* r=PyObject_CallObject(g_fn,a);
    Py_DECREF(a); Py_DECREF(xpy);
    if(!r){ PyErr_Print(); return -4; }
    at::Tensor rt=THPVariable_Unpack(r);
    if(rt.scalar_type()!=at::kBFloat16) rt=rt.to(at::kBFloat16);
    if(!rt.is_contiguous()) rt=rt.contiguous();
    // Do NOT memcpy into GRIMOIRE's buffer: at M=4096 that is 142 MB per call and
    // 192 calls per prefill = ~27 GB of pointless traffic, which costs far more than
    // the kernel saves.  vLLM uses the returned tensor directly; so do we.  The
    // tensor is kept alive in a small ring until the caller has consumed it.
    static PyObject* keep[8]={nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    static int slot=0;
    if(keep[slot]) Py_DECREF(keep[slot]);
    keep[slot]=r; slot=(slot+1)&7;
    *outptr=rt.data_ptr();
    return 0;
}

static PyObject *g_packed=nullptr;   // single-weight harness path
static int g_N=0,g_K=0;

// Bring up embedded CPython, import BesTLA, and pack ONE real checkpoint weight.
extern "C" int grimoire_bestla_init(int* N,int* K,int* G){
    if(!Py_IsInitialized()){ Py_Initialize(); PyRun_SimpleString("import sys; sys.argv=['grimoire']"); }
    if(!Py_IsInitialized()) return -1;
    if(!PyImport_ImportModule("torch")){ PyErr_Print(); return -2; }
    g_ark=PyImport_ImportModule("auto_round_kernel");
    if(!g_ark){ PyErr_Print(); return -3; }
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
      "DIMS=(N,K,G)\n";
    if(PyRun_SimpleString(prep)!=0){ PyErr_Print(); return -4; }
    PyObject* main=PyImport_AddModule("__main__");
    g_packed=PyObject_GetAttrString(main,"PACKED");
    PyObject* dims=PyObject_GetAttrString(main,"DIMS");
    if(!g_packed||!dims) return -5;
    g_N=(int)PyLong_AsLong(PyTuple_GetItem(dims,0));
    g_K=(int)PyLong_AsLong(PyTuple_GetItem(dims,1));
    g_G=(int)PyLong_AsLong(PyTuple_GetItem(dims,2));
    g_fn=PyObject_GetAttrString(g_ark,"woqgemm_linear");
    if(!g_fn) return -6;
    *N=g_N;*K=g_K;*G=g_G;
    return 0;
}

// x is a GRIMOIRE-owned USM bf16 buffer [M,K] on the caller's queue.
extern "C" int grimoire_bestla_linear(sycl::queue* q,const void* x,int M){
    c10::xpu::XPUStream ext=c10::xpu::getStreamFromExternal(q,0);
    c10::xpu::setCurrentXPUStream(ext);            // run BesTLA on GRIMOIRE's queue
    auto opts=at::TensorOptions().dtype(at::kBFloat16).device(at::kXPU);
    at::Tensor xt=at::from_blob(const_cast<void*>(x),{M,g_K},opts);
    PyObject* xpy=THPVariable_Wrap(xt);
    if(!xpy){ PyErr_Print(); return -1; }
    PyObject* a=Py_BuildValue("(OOOiiisssO)",xpy,g_packed,Py_None,g_N,g_K,g_G,
                              "bf16","int4","bf16",Py_False);
    PyObject* r=PyObject_CallObject(g_fn,a);
    Py_DECREF(a); Py_DECREF(xpy);
    if(!r){ PyErr_Print(); return -2; }
    Py_DECREF(r);
    return 0;
}
