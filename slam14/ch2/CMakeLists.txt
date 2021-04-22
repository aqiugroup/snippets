# 声明要求的 cmake 最低版本
# cmake_minimum_required( VERSION 2.8 )
cmake_minimum_required( VERSION 3.11 )

# 声明一个 cmake 工程
project( HelloSLAM )

# 设置编译模式
set( CMAKE_BUILD_TYPE "Debug" )

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 ）
# add_executable( helloSLAM helloSLAM.cpp )

# 添加一个库
add_library( hello libHelloSLAM.cpp )
# 共享库
add_library( hello_shared SHARED libHelloSLAM.cpp )

# 安装库到本地硬盘
INSTALL(TARGETS hello hello_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

INSTALL(FILES libHelloSLAM.h DESTINATION include/libHelloSLAM)

add_executable( useHello useHello.cpp )
# 将库文件链接到可执行程序上
target_link_libraries( useHello hello_shared )
