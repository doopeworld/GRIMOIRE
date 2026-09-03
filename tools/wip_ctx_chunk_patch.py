p = '/mnt/storage/isos/grimoire-fuse/src/grimoire.cpp'
s = open(p).read()

# 1. widen the context-ingest scratch buffers -------------------------------
old = """            dflash2.ctx=dfd(size_t(DM)*DH);
            if(cfg.is_muse){
                const size_t all_kv=size_t(dflash2.layers.size())*DM*DKV;"""
new = """            // Draft-context ingestion walks the whole prompt through fc +
            // context-KV. Doing that in DM=16 row chunks costs ~9 kernel
            // launches per chunk -- ~2250 launches for a 4k prompt, measured
            // at 285 ms and entirely launch-bound. The ingest is a plain
            // batched projection, so it can use a much wider chunk; only the
            // scratch has to grow with it.
            dflash2.ctx_chunk=[&]{
                const char* v=std::getenv("GRIMOIRE_DFLASH_CTX_CHUNK");
                int c=v&&*v?std::atoi(v):256;
                if(c<DM)c=DM;
                if(c>4096)c=4096;
                return c;
            }();
            const int DMC=std::max(DM,dflash2.ctx_chunk);
            dflash2.ctx=dfd(size_t(DMC)*DH);
            if(cfg.is_muse){
                const size_t all_kv=size_t(dflash2.layers.size())*DMC*DKV;"""
assert s.count(old) == 1, 'alloc anchor %d' % s.count(old)
s = s.replace(old, new)

# normed must also hold a full ingest chunk
old2 = """            dflash2.normed=dfd(size_t(DM)*DH);"""
new2 = """            dflash2.normed=dfd(size_t(DMC)*DH);"""
assert s.count(old2) == 1, 'normed anchor %d' % s.count(old2)
s = s.replace(old2, new2)

# 2. struct field -----------------------------------------------------------
old3 = """        bool fp16_draft=true;   // drafter weights uploaded as FP16"""
new3 = """        bool fp16_draft=true;   // drafter weights uploaded as FP16
        int  ctx_chunk=16;      // rows per draft-context ingest iteration"""
if old3 in s:
    s = s.replace(old3, new3)
else:
    old3b = """        float *conv_delta=nullptr, *conv_scratch=nullptr;"""
    assert s.count(old3b) == 1, 'struct anchor'
    s = s.replace(old3b, old3b + """
        int  ctx_chunk=16;      // rows per draft-context ingest iteration""")

# 3. use the wide chunk in the ingest loop ----------------------------------
old4 = """        const int start=dflash2.context_pos;
        const int rows=std::min(M,position-start);"""
new4 = """        const int start=dflash2.context_pos;
        const int rows=std::min(std::max(M,dflash2.ctx_chunk),position-start);"""
assert s.count(old4) == 1, 'loop anchor %d' % s.count(old4)
s = s.replace(old4, new4)

open(p, 'w').write(s)
print('context ingest chunk widened (GRIMOIRE_DFLASH_CTX_CHUNK, default 256)')
