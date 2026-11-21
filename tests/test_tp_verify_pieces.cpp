// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/tp_verify_pieces.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <thread>

using b70::tp_verify_pieces;
namespace {
constexpr std::size_t P = 10240, H = 5120, max = 65536;
struct alignas(64) Flag { std::atomic<std::uint64_t> ready{0}; };
Flag flag[2];
std::uint16_t shm[2][2][max];
std::uint16_t stage[2][max], received[2][max];
constexpr std::uint16_t payload(int rank, int seq, std::size_t lane) {
    return std::uint16_t((rank+2)*20000 + seq*127 + (lane*7 + (lane>>9))%1709);
}
void rank_thread(int rank) {
    const int peer = 1-rank;
    std::uint64_t counter = 0, legacy_counter = 0;
    constexpr int widths[] = {7,5,1,6,7,2,1,7,6,5,7,3,1};
    for(int s=0;s<2400;++s) {
        const std::size_t n = H*widths[s%std::size(widths)];
        const auto normal = tp_verify_pieces(n,P,false);
        const bool coalesce = normal.count>1 && s%3!=2;
        const auto plan = tp_verify_pieces(n,P,coalesce);
        assert(plan.count == (coalesce ? 1 : normal.count));
        legacy_counter += normal.count;
        for(std::size_t i=0;i<n;++i) stage[rank][i] = payload(rank,s,i);
        if(rank==(s&1)) std::this_thread::yield();
        for(std::size_t k=0;k<plan.count;++k) {
            const auto off=k*plan.size, count=std::min(plan.size,n-off);
            std::memcpy(shm[rank][s&1]+off,stage[rank]+off,count*sizeof(std::uint16_t));
            counter += plan.credit;
            flag[rank].ready.store(counter,std::memory_order_release);
            while(flag[peer].ready.load(std::memory_order_acquire)<counter) std::this_thread::yield();
            std::memcpy(received[rank]+off,shm[peer][s&1]+off,count*sizeof(std::uint16_t));
        }
        assert(counter==legacy_counter);
        for(std::size_t i=0;i<n;++i) assert(received[rank][i]==payload(peer,s,i));
    }
}
}
int main() {
    assert(tp_verify_pieces(0,P,true).count==0);
    assert(tp_verify_pieces(1,P,true).credit==1);
    assert(128*(19*tp_verify_pieces(7*H,P,false).count + tp_verify_pieces(6*H,P,false).count)==10112);
    assert(128*(19*tp_verify_pieces(7*H,P,true).count + tp_verify_pieces(6*H,P,true).count)==2560);
    for(auto n:{P+1,H*5,H*6,H*7,max}) {
        const auto old=tp_verify_pieces(n,P,false), alt=tp_verify_pieces(n,P,true);
        assert(old.size==P && alt.size==n && alt.count==1 && old.count==alt.credit);
    }
    std::thread a(rank_thread,0), b(rank_thread,1);
    a.join(); b.join();
    std::puts("PASS: 2400 mixed-width, skewed TP2 rendezvous; peer bytes and final legacy sequence counters exact (host only)");
}
