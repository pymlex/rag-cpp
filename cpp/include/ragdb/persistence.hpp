#pragma once

#include <string>

#include "ragdb/types.hpp"
#include "ragdb/vector_store.hpp"
#include "ragdb/link_store.hpp"

namespace ragdb {

bool persist_save(
    const std::string& path,
    const HnswParams& params,
    const VectorStore& store,
    const LinkStore& links);

bool persist_load(
    const std::string& path,
    HnswParams& params,
    VectorStore& store,
    LinkStore& links);

}