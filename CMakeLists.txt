cmake_minimum_required(VERSION 3.5.0)
project(parallel_bfs VERSION 0.1.0 LANGUAGES CXX)

find_package(Parlay REQUIRED)

set (CMAKE_CXX_STANDARD 20)

add_executable(parallel_bfs main.cpp)
target_link_libraries(parallel_bfs PRIVATE Parlay::parlay)