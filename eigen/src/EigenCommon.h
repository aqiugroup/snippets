#pragma once

#include <Eigen/Core>

#define EIGEN_FLOAT 1

#ifdef EIGEN_FLOAT

typedef Eigen::Matrix3f M3X;
typedef Eigen::Matrix4f M4X;

typedef Eigen::MatrixXf MXX;
typedef Eigen::VectorXf VXX;
typedef Eigen::VectorXi VXI;

#else

typedef Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > MXX;
typedef Eigen::Vector< float, Eigen::Dynamic, Eigen::Dynamic > VXX;

#endif
