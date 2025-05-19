cmake_minimum_required(VERSION 3.31.6)

project(regression VERSION 1.0 LANGUAGES CXX)

find_package(Matplot++ REQUIRED)

add_executable(model main.cpp)

target_link_libraries(model Matplot++::matplot)