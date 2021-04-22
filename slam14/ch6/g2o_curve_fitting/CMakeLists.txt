cmake_minimum_required( VERSION 2.8 )
project( g2o_curve_fitting )
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0002 NEW)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找G2O
# find_package( G2O REQUIRED )
set(G2O_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../../3rdparty/g2o)

#打印调试信息
MESSAGE(STATUS "_____________________________")
MESSAGE(STATUS "G2O_INCLUDE_DIRS: ${G2O_INCLUDE_DIRS}")
MESSAGE(STATUS "_____________________________")

include_directories(
    ${G2O_INCLUDE_DIRS}
    "/usr/local/include/eigen3"
)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( curve_fitting_g2o main.cpp )
# 与G2O和OpenCV链接
set(CMAKE_MACOSX_RPATH OFF)
target_link_libraries( curve_fitting_g2o
    ${OpenCV_LIBS}
    g2o_core g2o_stuff
)
