#include "ragdb/distance.hpp"

#include <cmath>
#include <algorithm>

namespace ragdb {

float inner_product(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float cosine_distance(const float* a, const float* b, std::size_t dim) {
    float ip = inner_product(a, b, dim);
    float d = 1.0f - ip;
    if (d < 0.0f) {
        d = 0.0f;
    }
    return d;
}

void normalize_inplace(std::vector<float>& v, std::size_t dim) {
    float norm = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        norm += v[i] * v[i];
    }
    norm = std::sqrt(norm);
    if (norm <= 1e-12f) {
        return;
    }
    for (std::size_t i = 0; i < dim; ++i) {
        v[i] /= norm;
    }
}

}