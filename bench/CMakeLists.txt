project(BlustBench)

add_executable(
    BlustBench
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops_bench.cc
)

target_link_libraries(BlustBench PUBLIC benchmark::benchmark blust)