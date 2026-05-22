cmake_minimum_required(VERSION 3.20)
project(ragdb_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
endif()

set(RAGDB_SOURCES
    src/candidate_queue.cpp
    src/distance.cpp
    src/hnsw_index.cpp
    src/hnsw_insert.cpp
    src/hnsw_search.cpp
    src/link_store.cpp
    src/neighbor_select.cpp
    src/persistence.cpp
    src/vector_store.cpp
    src/visited_set.cpp
)

add_library(ragdb_core STATIC ${RAGDB_SOURCES})
target_include_directories(ragdb_core PUBLIC include)

pybind11_add_module(ragdb_native bindings/ragdb_module.cpp)
target_link_libraries(ragdb_native PRIVATE ragdb_core)
target_include_directories(ragdb_native PRIVATE include)

if(WIN32)
    target_compile_definitions(ragdb_native PRIVATE NOMINMAX)
endif()
