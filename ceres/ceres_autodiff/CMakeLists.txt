cmake_minimum_required( VERSION 2.8 )
project( ceres_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

# 与Ceres和OpenCV链接
add_executable( ceres_autodiff_not_correct ceres_autodiff_not_correct_but_same_result.cpp )
target_link_libraries( ceres_autodiff_not_correct ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable( ceres_autodiff_correct ceres_autodiff_correct.cpp )
target_link_libraries( ceres_autodiff_correct ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable( ceres_autodiff_q1q2 ceres_autodiff_q1q2.cpp )
target_link_libraries( ceres_autodiff_q1q2 ${CERES_LIBRARIES} ${OpenCV_LIBS} )