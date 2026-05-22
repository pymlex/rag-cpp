#pragma once

#include <cmath>
#include <cstdint>
#include <vector>


namespace rag_db {

using Label = std::uint64_t;
using Level = int;

struct SearchHit {
    Label label;
    float distance;
};

struct Neighbor {
    Label label;
    float distance;
};

using Vector = std::vector<float>;

}
