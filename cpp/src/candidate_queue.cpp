#include "ragdb/candidate_queue.hpp"

namespace ragdb {

void CandidateMaxHeap::push(Id id, float dist) {
    heap_.push_back({id, dist});
    std::size_t i = heap_.size() - 1;
    while (i > 0) {
        std::size_t p = (i - 1) / 2;
        if (heap_[p].distance >= heap_[i].distance) {
            break;
        }
        std::swap(heap_[p], heap_[i]);
        i = p;
    }
}

bool CandidateMaxHeap::empty() const {
    return heap_.empty();
}

SearchHit CandidateMaxHeap::pop() {
    SearchHit out = heap_.front();
    heap_.front() = heap_.back();
    heap_.pop_back();
    std::size_t i = 0;
    while (true) {
        std::size_t l = 2 * i + 1;
        std::size_t r = 2 * i + 2;
        std::size_t largest = i;
        if (l < heap_.size() && heap_[l].distance > heap_[largest].distance) {
            largest = l;
        }
        if (r < heap_.size() && heap_[r].distance > heap_[largest].distance) {
            largest = r;
        }
        if (largest == i) {
            break;
        }
        std::swap(heap_[i], heap_[largest]);
        i = largest;
    }
    return out;
}

const SearchHit& CandidateMaxHeap::top() const {
    return heap_.front();
}

std::size_t CandidateMaxHeap::size() const {
    return heap_.size();
}

void CandidateMaxHeap::trim(std::size_t limit) {
    while (heap_.size() > limit) {
        pop();
    }
}

void CandidateMinHeap::push(Id id, float dist) {
    heap_.push_back({id, dist});
    std::size_t i = heap_.size() - 1;
    while (i > 0) {
        std::size_t p = (i - 1) / 2;
        if (heap_[p].distance <= heap_[i].distance) {
            break;
        }
        std::swap(heap_[p], heap_[i]);
        i = p;
    }
}

bool CandidateMinHeap::empty() const {
    return heap_.empty();
}

SearchHit CandidateMinHeap::pop() {
    SearchHit out = heap_.front();
    heap_.front() = heap_.back();
    heap_.pop_back();
    std::size_t i = 0;
    while (true) {
        std::size_t l = 2 * i + 1;
        std::size_t r = 2 * i + 2;
        std::size_t smallest = i;
        if (l < heap_.size() && heap_[l].distance < heap_[smallest].distance) {
            smallest = l;
        }
        if (r < heap_.size() && heap_[r].distance < heap_[smallest].distance) {
            smallest = r;
        }
        if (smallest == i) {
            break;
        }
        std::swap(heap_[i], heap_[smallest]);
        i = smallest;
    }
    return out;
}

const SearchHit& CandidateMinHeap::top() const {
    return heap_.front();
}

std::size_t CandidateMinHeap::size() const {
    return heap_.size();
}

}