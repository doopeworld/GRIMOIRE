// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef B70_TP_VERIFY_PIECES_HPP
#define B70_TP_VERIFY_PIECES_HPP
#include <cstddef>

namespace b70 {
struct TpVerifyPieces { std::size_t size, count, credit; };

// Credit advances the existing shared sequence counter, not the number of
// physical stores. A full-partial publish must end on the same sequence as
// the per-piece path, because unrelated reductions reuse that counter.
inline TpVerifyPieces tp_verify_pieces(std::size_t elems, std::size_t piece, bool whole) {
    if (!elems || !piece) return {};
    const auto count=1+(elems-1)/piece;
    return whole && count>1 ? TpVerifyPieces{elems,1,count} : TpVerifyPieces{piece,count,1};
}
}
#endif
