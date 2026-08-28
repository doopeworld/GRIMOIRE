#include "b70/native_model.hpp"
#include <cstdio>
int main(int argc,char**argv){
 if(argc<2||argc>3){std::fprintf(stderr,"usage: %s model.b70 [--list]\n",argv[0]);return 2;}
 b70::NativeModel m;std::string e;if(!m.open(argv[1],e)){std::fprintf(stderr,"%s\n",e.c_str());return 1;}
 auto&h=m.header();std::printf("MXFP4-GRIMOIRE v%u target bmg_g31 tensors %llu size %.2f GiB\n",
 h.version,(unsigned long long)h.tensor_count,double(h.file_size)/1073741824.0);
 if(argc==3) for(uint64_t i=0;i<h.tensor_count;++i){const auto&r=m.records()[i];
   std::printf("%s enc=%u rank=%u shape=",r.name,r.encoding,r.rank);
   for(uint32_t d=0;d<r.rank;++d)std::printf("%s%lld",d?"x":"",(long long)r.shape[d]);
   std::printf("\n");}
 return 0;
}
