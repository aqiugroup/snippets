project(QzcTest)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Debug)

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_HOME_DIRECTORY}/bin
    )

# enable_testing()
find_package(Threads REQUIRED)
# find_package(GTest REQUIRED)
# message(STATUS "GTest path: " ${GTEST_INCLUDE_DIR})
# include_directories(${GTEST_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/common)
add_subdirectory(common)

add_subdirectory(opencv)
add_subdirectory(eigen)