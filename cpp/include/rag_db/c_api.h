#pragma once

#include <stddef.h>
#include <stdint.h>


#ifdef _WIN32
#ifdef RAG_DB_BUILD
#define RAG_DB_API __declspec(dllexport)
#else
#define RAG_DB_API __declspec(dllimport)
#endif
#else
#define RAG_DB_API
#endif


#ifdef __cplusplus
extern "C" {
#endif

typedef struct RagDbHandle RagDbHandle;

RAG_DB_API RagDbHandle* rag_db_create(
    size_t dim,
    size_t max_elements,
    size_t m,
    size_t ef_construction,
    int cosine
);

RAG_DB_API void rag_db_destroy(RagDbHandle* handle);

RAG_DB_API uint64_t rag_db_add(RagDbHandle* handle, const float* vector);

RAG_DB_API void rag_db_mark_deleted(RagDbHandle* handle, uint64_t label);

RAG_DB_API size_t rag_db_search(
    RagDbHandle* handle,
    const float* query,
    size_t k,
    size_t ef,
    uint64_t* out_labels,
    float* out_distances
);

RAG_DB_API int rag_db_save(RagDbHandle* handle, const char* path);

RAG_DB_API int rag_db_load(RagDbHandle* handle, const char* path);

RAG_DB_API size_t rag_db_dim(RagDbHandle* handle);

RAG_DB_API uint64_t rag_db_size(RagDbHandle* handle);

#ifdef __cplusplus
}
#endif