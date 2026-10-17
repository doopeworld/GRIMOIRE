#!/usr/bin/env python3
"""Generate Unicode 13.0 mark/NFC tables for Qwen2's HF tokenizer.
Run with /tmp/gguf-golden-venv/bin/python (Python 3.9.6, Unicode 13.0.0).
"""
from pathlib import Path
import unicodedata as ud

if ud.unidata_version != '13.0.0':
    raise SystemExit('expected Unicode 13.0.0, got ' + ud.unidata_version)
root = Path(__file__).resolve().parents[1] / 'include' / 'b70'
marks=[]; start=prev=None
for cp in range(0x110000):
    if ud.category(chr(cp)).startswith('M'):
        if start is None: start=prev=cp
        elif cp==prev+1: prev=cp
        else: marks.append((start,prev));start=prev=cp
if start is not None:marks.append((start,prev))
with (root/'unicode_marks.hpp').open('w') as f:
    f.write('// Unicode 13.0.0 marks from Python 3.9.6 unicodedata. SPDX-License-Identifier: GPL-3.0-or-later\n#pragma once\n#include <cstdint>\n#include <cstddef>\nnamespace b70 {\nstruct UnicodeMarkRange { uint32_t first, last; };\nstatic constexpr UnicodeMarkRange kUnicodeMarks[] = {\n')
    for a,b in marks:f.write('    {0x%X, 0x%X},\n'%(a,b))
    f.write('};\ninline bool unicode_is_mark(uint32_t cp) {\n    size_t lo=0, hi=sizeof(kUnicodeMarks)/sizeof(kUnicodeMarks[0]);\n    while(lo<hi){const size_t mid=lo+(hi-lo)/2; if(kUnicodeMarks[mid].last<cp)lo=mid+1;else hi=mid;}\n    return lo<sizeof(kUnicodeMarks)/sizeof(kUnicodeMarks[0]) && kUnicodeMarks[lo].first<=cp;\n}\n}\n')
decomp=[];parts=[];classes=[];compositions=[]
for cp in range(0x110000):
    cc=ud.combining(chr(cp))
    if cc:classes.append((cp,cc))
    raw=ud.decomposition(chr(cp))
    if not raw or raw.startswith('<'):continue
    seq=[int(x,16) for x in raw.split()]
    decomp.append((cp,len(parts),len(seq)));parts+=seq
    if len(seq)==2 and ud.normalize('NFC',''.join(map(chr,seq)))==chr(cp):
        compositions.append((seq[0],seq[1],cp))
compositions.sort()
with (root/'unicode_nfc.hpp').open('w') as f:
    f.write('// Unicode 13.0.0 NFC data from Python 3.9.6 unicodedata. SPDX-License-Identifier: GPL-3.0-or-later\n#pragma once\n#include <cstdint>\n#include <cstddef>\n#include <string>\n#include <vector>\nnamespace b70 {\n')
    f.write('struct NFCDecomp { uint32_t cp, offset, len; };\nstatic constexpr NFCDecomp kNFCDecomp[] = {\n')
    for row in decomp:f.write('    {%d,%d,%d},\n'%row)
    f.write('};\nstatic constexpr uint32_t kNFCParts[] = {\n')
    for i in range(0,len(parts),16):f.write('    '+','.join(map(str,parts[i:i+16]))+',\n')
    f.write('};\nstruct NFCClass { uint32_t cp; uint8_t cc; };\nstatic constexpr NFCClass kNFCClasses[] = {\n')
    for row in classes:f.write('    {%d,%d},\n'%row)
    f.write('};\nstruct NFCCompose { uint32_t a,b,c; };\nstatic constexpr NFCCompose kNFCCompose[] = {\n')
    for row in compositions:f.write('    {%d,%d,%d},\n'%row)
    f.write('};\n')
    f.write('''inline const NFCDecomp* nfc_decomp(uint32_t cp) {
    size_t lo=0,hi=sizeof(kNFCDecomp)/sizeof(kNFCDecomp[0]);
    while(lo<hi){size_t m=lo+(hi-lo)/2;if(kNFCDecomp[m].cp<cp)lo=m+1;else hi=m;}
    return lo<sizeof(kNFCDecomp)/sizeof(kNFCDecomp[0]) && kNFCDecomp[lo].cp==cp ? &kNFCDecomp[lo] : nullptr;
}
inline uint8_t nfc_class(uint32_t cp) {
    size_t lo=0,hi=sizeof(kNFCClasses)/sizeof(kNFCClasses[0]);
    while(lo<hi){size_t m=lo+(hi-lo)/2;if(kNFCClasses[m].cp<cp)lo=m+1;else hi=m;}
    return lo<sizeof(kNFCClasses)/sizeof(kNFCClasses[0]) && kNFCClasses[lo].cp==cp ? kNFCClasses[lo].cc : 0;
}
inline uint32_t nfc_compose(uint32_t a,uint32_t b) {
    // Hangul canonical composition (Unicode UAX #15 D117).
    if(a>=0x1100&&a<0x1113&&b>=0x1161&&b<0x1176)
        return 0xAC00+((a-0x1100)*21+(b-0x1161))*28;
    if(a>=0xAC00&&a<=0xD7A3&&(a-0xAC00)%28==0&&b>=0x11A8&&b<0x11C3)
        return a+(b-0x11A7);
    size_t lo=0,hi=sizeof(kNFCCompose)/sizeof(kNFCCompose[0]);
    while(lo<hi){size_t m=lo+(hi-lo)/2;const auto &x=kNFCCompose[m];
        if(x.a<a||(x.a==a&&x.b<b))lo=m+1;else hi=m;}
    if(lo<sizeof(kNFCCompose)/sizeof(kNFCCompose[0])&&
       kNFCCompose[lo].a==a&&kNFCCompose[lo].b==b)return kNFCCompose[lo].c;
    return 0;
}
inline void nfc_expand(uint32_t cp,std::vector<uint32_t>& out) {
    if(cp>=0xAC00&&cp<=0xD7A3){
        uint32_t s=cp-0xAC00;out.push_back(0x1100+s/588);
        out.push_back(0x1161+(s%588)/28);
        if(s%28)out.push_back(0x11A7+s%28);
    }else if(const NFCDecomp* d=nfc_decomp(cp)){
        for(uint32_t i=0;i<d->len;++i)nfc_expand(kNFCParts[d->offset+i],out);
    }else out.push_back(cp);
}
inline std::string unicode_nfc(const std::string& text) {
    std::vector<uint32_t> decomposed;decomposed.reserve(text.size());
    for(size_t i=0;i<text.size();){
        const uint8_t c=uint8_t(text[i]);uint32_t cp=c;size_t n=1;
        if((c&0xE0)==0xC0&&i+1<text.size()){cp=c&31;n=2;}
        else if((c&0xF0)==0xE0&&i+2<text.size()){cp=c&15;n=3;}
        else if((c&0xF8)==0xF0&&i+3<text.size()){cp=c&7;n=4;}
        for(size_t j=1;j<n;++j)cp=(cp<<6)|(uint8_t(text[i+j])&63);
        nfc_expand(cp,decomposed);i+=n;
    }
    // Canonical order: stable insertion of combining marks after starter.
    for(size_t i=1;i<decomposed.size();++i){
        const uint8_t cc=nfc_class(decomposed[i]);
        if(!cc)continue;
        size_t j=i;
        while(j>0){const uint8_t prev=nfc_class(decomposed[j-1]);
            if(!prev||prev<=cc)break;
            const uint32_t t=decomposed[j];decomposed[j]=decomposed[j-1];decomposed[j-1]=t;--j;}
    }
    std::vector<uint32_t> composed;composed.reserve(decomposed.size());
    size_t starter=0;uint8_t last_cc=0;
    for(uint32_t cp:decomposed){
        const uint8_t cc=nfc_class(cp);
        if(!composed.empty()&&(last_cc<cc||last_cc==0)){
            if(uint32_t merged=nfc_compose(composed[starter],cp)){
                composed[starter]=merged;continue;
            }
        }
        if(cc==0)starter=composed.size();
        composed.push_back(cp);last_cc=cc;
    }
    std::string out;out.reserve(text.size());
    for(uint32_t cp:composed){
        if(cp<0x80)out.push_back(char(cp));
        else if(cp<0x800){out.push_back(char(0xC0|(cp>>6)));out.push_back(char(0x80|(cp&63)));}
        else if(cp<0x10000){out.push_back(char(0xE0|(cp>>12)));out.push_back(char(0x80|((cp>>6)&63)));out.push_back(char(0x80|(cp&63)));}
        else {out.push_back(char(0xF0|(cp>>18)));out.push_back(char(0x80|((cp>>12)&63)));out.push_back(char(0x80|((cp>>6)&63)));out.push_back(char(0x80|(cp&63)));}
    }
    return out;
}
} // namespace b70
''')
print('marks',len(marks),'decompositions',len(decomp),'classes',len(classes),'compositions',len(compositions))
