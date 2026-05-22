#pragma once

#include <cmath>
#include <cstdlib>
#include <random>

#include "rag_db/config.hpp"


namespace rag_db {

class LevelGenerator {
public:
    explicit LevelGenerator(float ml = kDefaultMl, unsigned seed = 42)
        : ml_(ml), rng_(seed), uniform_(0.0f, 1.0f) {}

    int sample_level() {
        float r = uniform_(rng_);
        if (r <= 0.0f) {
            r = 1e-6f;
        }
        int level = static_cast<int>(-std::log(r) * ml_);
        if (level > static_cast<int>(kMaxLevelCap)) {
            level = static_cast<int>(kMaxLevelCap);
        }
        return level;
    }

private:
    float ml_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_;
};

}