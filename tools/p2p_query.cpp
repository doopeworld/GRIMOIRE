// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// p2p_query.cpp -- read-only: can the two B70s reach each other's memory (P2P)?
// Device queries only: no context, no allocation, no transfer.
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    std::printf("%zu GPU(s)\n", devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        std::string pci = devs[i].has(sycl::aspect::ext_intel_pci_address)
            ? devs[i].get_info<sycl::ext::intel::info::device::pci_address>() : "?";
        std::printf("  [%zu] %s %s\n", i, devs[i].get_info<sycl::info::device::name>().c_str(), pci.c_str());
    }
    namespace exp = sycl::ext::oneapi;
    for (size_t i = 0; i < devs.size(); ++i)
        for (size_t j = 0; j < devs.size(); ++j) if (i != j) {
            bool acc = devs[i].ext_oneapi_can_access_peer(devs[j], exp::peer_access::access_supported);
            bool atom = devs[i].ext_oneapi_can_access_peer(devs[j], exp::peer_access::atomics_supported);
            std::printf("  %zu -> %zu: peer access %s, peer atomics %s\n", i, j, acc ? "YES" : "no", atom ? "YES" : "no");
        }
    return 0;
}
