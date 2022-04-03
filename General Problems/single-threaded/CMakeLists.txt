cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O3 -std=c++11")
project(ex1)

add_executable(
    ${PROJECT_NAME} ex1.c
)
project(ex2)

add_executable(
    ${PROJECT_NAME} ex2.c
)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    -lm
)
