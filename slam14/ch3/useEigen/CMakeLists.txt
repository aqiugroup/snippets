cmake_minimum_required(VERSION 3.10)
project(useEigen)

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-g" )

# 添加Eigen头文件
# include_directories( "/usr/local/include/eigen3" )

# in osx and brew install
# include_directories( /usr/local/Cellar/eigen/3.3.7/include/eigen3 )

include_directories( include )

add_library( libHello src/Hello.cpp )
add_executable( eigenMatrix eigenMatrix.cpp )

target_link_libraries( eigenMatrix libHello )
