#pragma once

#include <vector>
#include <algorithm>

#include "ragdb/types.hpp"

namespace ragdb {

class CandidateMaxHeap {
public:
    void push(Id id, float dist);

    bool empty() const;

    SearchHit pop();

    const SearchHit& top() const;

    std::size_t size() const;

    void trim(std::size_t limit);

private:
    std::vector<SearchHit> heap_;
};

class CandidateMinHeap {
public:
    void push(Id id, float dist);

    bool empty() const;

    SearchHit pop();

    const SearchHit& top() const;

    std::size_t size() const;

private:
    std::vector<SearchHit> heap_;
};

}