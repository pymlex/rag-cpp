#pragma once

#include <cmath>
#include <cstddef>


namespace rag_db {

inline float l2_squared(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}


inline float inner_product(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}


inline float cosine_distance(const float* a, const float* b, std::size_t dim) {
    float dot = inner_product(a, b, dim);
    float na = 0.0f;
    float nb = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom <= 1e-12f) {
        return 1.0f;
    }
    return 1.0f - dot / denom;
}

}