cmake_minimum_required(VERSION 3.31.6)

project(regression VERSION 1.0 LANGUAGES CXX)

find_package(Matplot++ REQUIRED)

add_executable(model main.cpp)

add_executable(logistic_model logistic_main.cpp)

target_include_directories(model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(model Matplot++::matplot)

target_link_libraries(logistic_model Matplot++::matplot)