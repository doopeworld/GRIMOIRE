import io, sys
p = "/grimoire/src/grimoire.cpp"
s = io.open(p, encoding="utf-8").read()
orig = s

anchor = """    auto dense=load_xe2_dense_mxfp4_f32();
    auto mm_f16_raw=[&](const DevQuant& w,const float* x,int rows){"""
assert s.count(anchor) == 1, "anchor count %d" % s.count(anchor)

helpers = """    const char* dump_dir=std::getenv("GRIMOIRE_DFLASH_DUMP");
    static int dflash_dump_call=0;
    const bool dumping=dump_dir&&*dump_dir&&dflash_dump_call==0;
    ++dflash_dump_call;
    auto dump_write=[&](const std::string& name,const std::vector<float>& h){
        std::string fn=std::string(dump_dir)+"/g_"+name+".f32";
        std::FILE* f=std::fopen(fn.c_str(),"wb");
        if(!f){std::fprintf(stderr,"  dflash dump: cannot open %s\\n",fn.c_str());return;}
        std::fwrite(h.data(),sizeof(float),h.size(),f);
        std::fclose(f);
        std::fprintf(stderr,"  dflash dump: %s [%zu]\\n",name.c_str(),h.size());
    };
    auto dump_f32=[&](const std::string& name,const float* p,size_t n){
        if(!dumping||!p||!n)return;
        std::vector<float> h(n);
        q.memcpy(h.data(),p,n*sizeof(float)).wait();
        dump_write(name,h);
    };
    auto dump_f16=[&](const std::string& name,const sycl::half* p,size_t n){
        if(!dumping||!p||!n)return;
        std::vector<sycl::half> hh(n);
        q.memcpy(hh.data(),p,n*sizeof(sycl::half)).wait();
        std::vector<float> h(n);
        for(size_t i=0;i<n;++i)h[i]=float(hh[i]);
        dump_write(name,h);
    };
"""
s = s.replace(anchor, helpers + anchor, 1)

# stage 1/2/3: inside the context loop
a2 = """        fc_mm(dflash2.target_aux+int64_t(start)*NT*H,dflash2.ctx,rows);
        norm(dflash2.ctx,nullptr,dflash2.hidden_norm,dflash2.hidden_norm_f16,
             dflash2.normed,rows);"""
assert s.count(a2) == 1
s = s.replace(a2, """        dump_f32("01_aux_"+std::to_string(start),
                 dflash2.target_aux+int64_t(start)*NT*H,size_t(rows)*NT*H);
""" + a2 + """
        dump_f32("02_fc_"+std::to_string(start),dflash2.ctx,size_t(rows)*H);
        dump_f32("03_ctxnorm_"+std::to_string(start),dflash2.normed,size_t(rows)*H);""", 1)

a3 = """            launch_dflash_context_kv_f16w(q,dflash2.context_kv_all,
                dflash2.context_k_all_f16,dflash2.context_v_all_f16,
                dflash2.k_norm_all_f16,int(dflash2.layers.size()),rows,KVH,HD,
                start,theta,eps,{});"""
assert s.count(a3) == 1
s = s.replace(a3, """            dump_f32("04_ctxkv_"+std::to_string(start),dflash2.context_kv_all,
                     size_t(rows)*dflash2.fused_context_kv.w.N);
""" + a3 + """
            dump_f16("05_ctxk_"+std::to_string(start),dflash2.context_k_all_f16,
                     dflash2.layers.size()*size_t(rows)*KVW);
            dump_f16("06_ctxv_"+std::to_string(start),dflash2.context_v_all_f16,
                     dflash2.layers.size()*size_t(rows)*KVW);""", 1)

# stage 4: block embedding
a4 = """    launch_embed_f16_batched(q,dflash2.shared_embed_f16.fp16,dflash2.tokens,
                             dflash2.resid,M,H);
    checkpoint("block embedding");"""
assert s.count(a4) == 1
s = s.replace(a4, a4 + """
    dump_f32("07_blockembed",dflash2.resid,size_t(M)*H);""", 1)

# stage 5: per-layer
a5 = """        if(cfg.is_muse){
            const sycl::half* qkv=mm_f16_raw(d.qkv,dflash2.normed,M);
            launch_qkv_norm_rope_f16w_fused(q,qkv,dflash2.q_f16,
                dflash2.k_f16,dflash2.v_f16,d.q_norm_f16,d.k_norm_f16,M,QH,KVH,HD,
                position,theta,eps,{});"""
assert s.count(a5) == 1
s = s.replace(a5, """        dump_f32("08_L"+std::to_string(li)+"_innorm",dflash2.normed,size_t(M)*H);
""" + a5 + """
            dump_f16("09_L"+std::to_string(li)+"_qkv",qkv,
                     size_t(M)*d.qkv.w.N);
            dump_f16("10_L"+std::to_string(li)+"_q",dflash2.q_f16,size_t(M)*QW);
            dump_f16("11_L"+std::to_string(li)+"_k",dflash2.k_f16,size_t(M)*KVW);
            dump_f16("12_L"+std::to_string(li)+"_v",dflash2.v_f16,size_t(M)*KVW);""", 1)

a6 = """        mm(d.o,dflash2.attn,dflash2.proj,M);"""
assert s.count(a6) == 1
s = s.replace(a6, """        dump_f32("13_L"+std::to_string(li)+"_attn",dflash2.attn,size_t(M)*QW);
""" + a6 + """
        dump_f32("14_L"+std::to_string(li)+"_o",dflash2.proj,size_t(M)*H);""", 1)

a7 = """        mm(d.down,dflash2.h,dflash2.mlp,M);
        checkpoint("MLP");"""
assert s.count(a7) == 1
s = s.replace(a7, a7 + """
        dump_f32("15_L"+std::to_string(li)+"_mlp",dflash2.mlp,size_t(M)*H);""", 1)

# stage 6: final norm + logits
a8 = """    norm(dflash2.resid,dflash2.mlp,dflash2.norm,dflash2.norm_f16,
         dflash2.normed,M);

    auto w4=load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32_m16");"""
assert s.count(a8) == 1
s = s.replace(a8, a8 + """
    dump_f32("16_finalnorm",dflash2.normed,size_t(M)*H);""", 1)

a9 = """    draft_tokens.resize(M-1);
    q.memcpy(draft_tokens.data(),dflash2.draft_ids,
             size_t(M-1)*sizeof(int32_t)).wait();"""
assert s.count(a9) == 1
s = s.replace(a9, """    dump_f32("17_logits",dflash2.logits,size_t(M-1)*cfg.vocab);
""" + a9 + """
    if(dumping){
        std::fprintf(stderr,"  dflash dump: draft_ids");
        for(int r=0;r<M-1;++r)
            std::fprintf(stderr," %d",draft_tokens[size_t(r)]);
        std::fprintf(stderr,"\\n");
    }""", 1)

assert s != orig
io.open(p,"w",encoding="utf-8").write(s)
print("patched OK")
