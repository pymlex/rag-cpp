#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>

namespace ragdb {

using Id = std::int64_t;
using Level = std::int32_t;

constexpr Id kInvalidId = -1;

struct HnswParams {
    std::size_t dim = 312;
    std::size_t max_elements = 200000;
    std::int32_t M = 16;
    std::int32_t M_max0 = 32;
    std::int32_t ef_construction = 200;
    std::int32_t ef_search = 64;
    float level_mult = 1.0f / std::log(static_cast<float>(M));
};

struct SearchHit {
    Id id;
    float distance;
};

}