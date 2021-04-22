# 声明要求的 cmake 最低版本
# cmake_minimum_required( VERSION 2.8 )
cmake_minimum_required( VERSION 3.11 )
# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11")
set( CMAKE_CXX_FLAGS "-g" )

# 声明一个 cmake 工程
project( ax_b )

# 设置编译模式
set( CMAKE_BUILD_TYPE "Debug" )

include_directories( /usr/local/Cellar/eigen/3.3.4/include/eigen3 )

add_executable( ax_b Ax_b.cpp ) 
# 将库文件链接到可执行程序上
# target_link_libraries( )
