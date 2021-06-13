aux_source_directory(. EXECUTABLE)
set(CMAKE_BUILD_TYPE Debug)

# 添加Eigen头文件
# ubuntu
# include_directories( "/usr/include/eigen3" )
# macos
include_directories( "/usr/local/include/eigen3" )


add_subdirectory(trackBall)



