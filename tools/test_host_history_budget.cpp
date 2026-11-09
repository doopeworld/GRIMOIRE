// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/host_history_budget.hpp"
#include <cstdio>
int main() {
    using b70::host_history_slot_limit;
    constexpr uint64_t gib=uint64_t(1)<<30;
    // Two B70 ranks, 131K history=12.5GiB each; first histories exist already.
    if(host_history_slot_limit(70*gib,8*gib,25*gib/2,2,8)!=3)return 1;
    // A single-rank deployment can fit more slots in the same remaining RAM.
    if(host_history_slot_limit(70*gib,8*gib,25*gib/2,1,8)!=5)return 2;
    // Exactly enough for one extra pair, then one byte below that boundary.
    if(host_history_slot_limit(33*gib,8*gib,25*gib/2,2,8)!=2)return 3;
    if(host_history_slot_limit(33*gib-1,8*gib,25*gib/2,2,8)!=1)return 4;
    if(host_history_slot_limit(7*gib,8*gib,25*gib/2,2,8)!=1)return 5;
    if(host_history_slot_limit(100*gib,8*gib,25*gib/2,2,1)!=1)return 6;
    if(host_history_slot_limit(100*gib,8*gib,0,2,8)!=8)return 7;
    if(host_history_slot_limit(UINT64_MAX,0,UINT64_MAX,2,8)!=1)return 8;
    std::puts("HOST_HISTORY_BUDGET_PASS");
}
