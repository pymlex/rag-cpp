#pragma once

#include <cstdint>
#include <vector>
#include <mutex>

#include "ragdb/types.hpp"

namespace ragdb {

class VisitedSet {
public:
    explicit VisitedSet(std::size_t capacity);

    bool mark(Id id);

    void reset();

private:
    std::vector<std::uint32_t> stamps_;
    std::vector<std::uint32_t> tags_;
    std::uint32_t generation_;
    std::mutex mutex_;
};

}