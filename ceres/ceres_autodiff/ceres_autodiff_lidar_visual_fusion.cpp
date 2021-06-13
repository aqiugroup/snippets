#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/internal/autodiff.h>
#include <ceres/internal/numeric_diff.h>
#include <cstdlib>
#include <ctime>

// goal : use numeric diff and auto diff to check my analytic diff.
// requirement : eigen>3.3.5 ceres==1.14.0
// target function : q_LI * q_Il1_Ir * q_Il2_Ir.conjugate() * q_LI.conjugate()

class QuaternionCostFunctor {
public:
  QuaternionCostFunctor(const Eigen::Quaterniond &_q_LI, const Eigen::Vector3d &_p_LI,
                        const Eigen::Quaterniond &_q_L1_L2_meas, const Eigen::Vector3d &_p_L1_L2_meas) :
    q_LI_(_q_LI), p_LI_(_p_LI), q_L1_L2_meas_(_q_L1_L2_meas), p_L1_L2_meas_(_p_L1_L2_meas) {}

  template <typename T>
  bool operator()(const T *const _q_Il1_Ir, const T *const _q_Il2_Ir,
                  T *_e) const {
    const Eigen::Quaternion<T> q_Il1_Ir(_q_Il1_Ir);
    const Eigen::Quaternion<T> q_Il2_Ir(_q_Il2_Ir);

    const Eigen::Quaternion<T> q_LI(
        static_cast<T>(q_LI_.w()), static_cast<T>(q_LI_.x()),
        static_cast<T>(q_LI_.y()), static_cast<T>(q_LI_.z()));
    const Eigen::Matrix<T, 3, 1> p_LI(
        static_cast<T>(p_LI_.x()), static_cast<T>(p_LI_.y()),
        static_cast<T>(p_LI_.z()));
    const Eigen::Quaternion<T> q_L1_L2_meas(
        static_cast<T>(q_L1_L2_meas_.w()), static_cast<T>(q_L1_L2_meas_.x()),
        static_cast<T>(q_L1_L2_meas_.y()), static_cast<T>(q_L1_L2_meas_.z()));
    const Eigen::Matrix<T, 3, 1> p_L1_L2_meas(
        static_cast<T>(p_L1_L2_meas_.x()), static_cast<T>(p_L1_L2_meas_.y()),
        static_cast<T>(p_L1_L2_meas_.z()));


    const Eigen::Quaternion<T> q_L1_L2_pred = q_LI * q_Il1_Ir * q_Il2_Ir.conjugate() * q_LI.conjugate();
    // const Eigen::Matrix<T, 3, 1> p_L1_L2_pred = q_LI * (q_Il1_Ir * (p_Ir_Il2 - p_Ir_Il1)) + p_LI - q_L1_L2_pred * p_LI;

    const Eigen::Quaternion<T> e_q = q_L1_L2_pred * q_L1_L2_meas.inverse();
    const Eigen::AngleAxis<T> e_aa(e_q);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> e(_e);
    e = e_aa.axis() * e_aa.angle();

    return true;
  }

  void evaluateAnalytically(const double *const _q_Il1_Ir,
                            const double *const _q_Il2_Ir,
                            double *_e,
                            double **_jacobians) const {
    const Eigen::Quaternion<double> q_Il1_Ir(_q_Il1_Ir);
    const Eigen::Quaternion<double> q_Il2_Ir(_q_Il2_Ir);

    const Eigen::Quaterniond q_LI = q_LI_;
    const Eigen::Vector3d p_LI = p_LI_;

    const Eigen::Quaterniond q_L1_L2_pred = q_LI * q_Il1_Ir * q_Il2_Ir.conjugate() * q_LI.conjugate();
    // Eigen::Vector3d p_L1_L2_pred = q_LI * (q_Il1_Ir * (p_Ir_Il2 - p_Ir_Il1)) + p_LI - q_L1_L2_meas * p_LI;

    const Eigen::Quaternion<double> e_q = q_L1_L2_pred * q_L1_L2_meas_.inverse();
    const Eigen::AngleAxis<double> e_aa(e_q);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> e(_e);
    e = e_aa.axis() * e_aa.angle();

    if (_jacobians != nullptr) {
      if (_jacobians[0] != nullptr) { /// q_Il1_Ir
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
            _jacobians[0]);
        J.setZero();
        J.block(0, 0, 3, 3) = q_LI_.toRotationMatrix();
      }
      if (_jacobians[1] != nullptr) { /// q_Il2_Ir
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
            _jacobians[1]);
        J.setZero();
        J.block(0, 0, 3, 3) = -(q_LI_ * q_Il1_Ir * q_Il2_Ir.conjugate()).toRotationMatrix();
      }
    }
  }

  // 右扰动
  static bool GlobalToLocal(const double *x, double *jacobian) {
    const double qw = x[3];
    const double qx = x[0];
    const double qy = x[1];
    const double qz = x[2];
    jacobian[0] = qw, jacobian[1] = qz, jacobian[2] = -qy;
    jacobian[3] = -qz, jacobian[4] = qw, jacobian[5] = qx;
    jacobian[6] = qy, jacobian[7] = -qx, jacobian[8] = qw;
    jacobian[9] = -qx, jacobian[10] = -qy, jacobian[11] = -qz;
    return true;
  }

  static inline Eigen::Matrix3d skew(const Eigen::Vector3d &_v) {
    Eigen::Matrix3d res;
    res.setZero();
    res(0, 1) = -_v[2], res(0, 2) = _v[1], res(1, 2) = -_v[0];
    res(1, 0) = _v[2], res(2, 0) = -_v[1], res(2, 1) = _v[0];
    return res;
  }

  static Eigen::Matrix3d J_l(const double _angle,
                             const Eigen::Vector3d &_axis) {
    /// (145)
    Eigen::Matrix3d res =
        Eigen::Matrix3d::Identity() +
        (1 - std::cos(_angle)) / (_angle * _angle) * skew(_angle * _axis) +
        (_angle - std::sin(_angle)) / (_angle * _angle * _angle) *
            skew(_angle * _axis) * skew(_angle * _axis);
    return res;
  }

  static Eigen::Matrix3d J_l_inv(const double _angle,
                                 const Eigen::Vector3d &_axis) {
    /// (146)
    Eigen::Matrix3d res =
        Eigen::Matrix3d::Identity() - 0.5 * skew(_angle * _axis) +
        (1 / (_angle * _angle) -
         (1 + std::cos(_angle)) / (2 * _angle * std::sin(_angle))) *
            skew(_angle * _axis) * skew(_angle * _axis);
    return res;
  }

  static Eigen::Matrix3d J_r(const double _angle,
                             const Eigen::Vector3d &_axis) {
    return J_l(_angle, _axis).transpose();
    /// (143)
    Eigen::Matrix3d res =
        Eigen::Matrix3d::Identity() -
        (1 - std::cos(_angle)) / (_angle * _angle) * skew(_angle * _axis) +
        (_angle - std::sin(_angle)) / (_angle * _angle * _angle) *
            skew(_angle * _axis) * skew(_angle * _axis);
    return res;
  }

  static Eigen::Matrix3d J_r_inv(const double _angle,
                                 const Eigen::Vector3d &_axis) {
    return J_l_inv(_angle, _axis).transpose();
    /// (144)
    Eigen::Matrix3d res =
        Eigen::Matrix3d::Identity() + 0.5 * skew(_angle * _axis) +
        (1 / (_angle * _angle) -
         (1 + std::cos(_angle)) / (2 * _angle * std::sin(_angle))) *
            skew(_angle * _axis) * skew(_angle * _axis);
    return res;
  }

private:
  const Eigen::Quaterniond q_L1_L2_meas_;
  const Eigen::Vector3d p_L1_L2_meas_;
  const Eigen::Quaterniond q_LI_;
  const Eigen::Vector3d p_LI_;
  // Eigen::Quaternion<T> q_L1_L2_meas_;
  // Eigen::Matrix<T, 3, 1> p_L1_L2_meas_;
  // Eigen::Quaternion<T> q_LI;
  // Eigen::Matrix<T, 3, 1> p_LI;
};

Eigen::Quaterniond getRandomQuaternion() {
  const double range = 1.;

  Eigen::Vector3d axis(std::rand() / double(RAND_MAX) * 2 * range + (-range),
                       std::rand() / double(RAND_MAX) * 2 * range + (-range),
                       std::rand() / double(RAND_MAX) * 2 * range + (-range));
  axis.normalize();
  const double angle = std::rand() / double(RAND_MAX) * 2 * M_PI;
  Eigen::AngleAxisd aa(angle, axis);

  return Eigen::Quaterniond(aa);
}

// target function : q_Il2_Ir * q_Il1_Ir.inverse()
int main(int argc, char **argv) {
  std::srand(std::time(NULL));
  std::srand(0);

  // double *jac = new double[9];
  // jac[0]=1, jac[4]=1, jac[8]=1, jac[1]=-1;
  // Eigen::Vector3d p(1,1,1);
  // Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J1(jac);
  // Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> J2(jac);
  // std::cout << " j1\n" << J1 << std::endl;
  // std::cout << " j1 " << (J1 * p).transpose() << " J1(1) " << J1(1) << std::endl;
  // std::cout << " j2\n" << J2 << std::endl;
  // std::cout << " j2 " << (J2 * p).transpose() << " J2(1) " << J2(1)<< std::endl;

  const double range = 1.;
  Eigen::Quaterniond q_LI = getRandomQuaternion();
  Eigen::Vector3d p_LI = Eigen::Vector3d(std::rand() / double(RAND_MAX) * 2 * range + (-range),
                                          std::rand() / double(RAND_MAX) * 2 * range + (-range),
                                          std::rand() / double(RAND_MAX) * 2 * range + (-range));

  Eigen::Quaterniond q_Il1_Ir = getRandomQuaternion();
  Eigen::Quaterniond q_Il2_Ir = getRandomQuaternion();
  Eigen::Vector3d p_Ir_Il1 = Eigen::Vector3d(std::rand() / double(RAND_MAX) * 2 * range + (-range),
                                             std::rand() / double(RAND_MAX) * 2 * range + (-range),
                                             std::rand() / double(RAND_MAX) * 2 * range + (-range));
  Eigen::Vector3d p_Ir_Il2 = Eigen::Vector3d(std::rand() / double(RAND_MAX) * 2 * range + (-range),
                                             std::rand() / double(RAND_MAX) * 2 * range + (-range),
                                             std::rand() / double(RAND_MAX) * 2 * range + (-range));
  // std::cout << "q_Il1_Ir:\n" << q_Il1_Ir.toRotationMatrix() << std::endl;
  // std::cout << "q_Il1_Ir:\n" << q_Il1_Ir.toRotationMatrix() << std::endl;
  std::cout << "p_Ir_Il1: " << p_Ir_Il1.transpose() << std::endl;
  std::cout << "p_Ir_Il2: " << p_Ir_Il2.transpose() << std::endl;

  const Eigen::Quaterniond q_L1_L2_meas = q_LI * q_Il1_Ir * q_Il2_Ir.conjugate() * q_LI.conjugate();
  Eigen::Vector3d p_L1_L2_meas = q_LI * (q_Il1_Ir * (p_Ir_Il2 - p_Ir_Il1)) + p_LI - q_L1_L2_meas * p_LI;
  std::cout << "q_L1_L2_meas:\n" << q_L1_L2_meas.toRotationMatrix() << std::endl;
  std::cout << "p_L1_L2_meas:\n" << p_L1_L2_meas.transpose() << std::endl;

  QuaternionCostFunctor functor(q_LI, p_LI, q_L1_L2_meas, p_L1_L2_meas);

  double residuals[3];
  double *parameters[2] = {q_Il1_Ir.coeffs().data(), q_Il2_Ir.coeffs().data()};
  double **jacobians = new double *[2];
  for (int i = 0; i < 2; ++i)
    jacobians[i] = new double[12];

  {
    std::cout << "----------------------------------------\n";
    ceres::internal::AutoDiff<QuaternionCostFunctor, double, 4, 4
                              >::Differentiate(functor, parameters,
                                                3, /// residual num
                                                residuals, jacobians);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(
        jacobians[0]);

    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_0;
    QuaternionCostFunctor::GlobalToLocal(parameters[0],
                                         global_to_local_0.data());

    std::cout << "autodiff jacobian_0:\n"
              << 0.5 * jacobian_0 * global_to_local_0 << std::endl;

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(
        jacobians[1]);

    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_1;
    QuaternionCostFunctor::GlobalToLocal(parameters[1],
                                         global_to_local_1.data());

    std::cout << "autodiff jacobian_1:\n"
              << 0.5 * jacobian_1 * global_to_local_1 << std::endl;
  }

  {
    std::cout << "----------------------------------------\n";
    ceres::internal::NumericDiff<
        QuaternionCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 3, 4, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        4>::EvaluateJacobianForParameterBlock(&functor, residuals,
                                              ceres::NumericDiffOptions(),
                                              3, /// residual num
                                              0, /// block index
                                              4, /// block size
                                              parameters, jacobians[0]);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(
        jacobians[0]);

    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_0;
    QuaternionCostFunctor::GlobalToLocal(parameters[0],
                                         global_to_local_0.data());

    std::cout << "numdiff jacobian_0:\n"
              << 0.5 * jacobian_0 * global_to_local_0 << std::endl;
  }

  {
    ceres::internal::NumericDiff<
        QuaternionCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 3, 4, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 1,
        4>::EvaluateJacobianForParameterBlock(&functor, residuals,
                                              ceres::NumericDiffOptions(),
                                              3, /// residual num
                                              1, /// block index
                                              4, /// block size
                                              parameters, jacobians[1]);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(
        jacobians[1]);

    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_1;
    QuaternionCostFunctor::GlobalToLocal(parameters[1],
                                         global_to_local_1.data());

    std::cout << "numdiff jacobian_1:\n"
              << 0.5 * jacobian_1 * global_to_local_1 << std::endl;
  }

  {
    functor.evaluateAnalytically(parameters[0], parameters[1],
                                 residuals, jacobians);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(
        jacobians[0]);
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(
        jacobians[1]);

    std::cout << "----------------------------------------\n";
    std::cout << "analytic jacobian_0:\n"
              << jacobian_0.block(0, 0, 3, 3) << std::endl;
    std::cout << "analytic jacobian_1:\n"
              << jacobian_1.block(0, 0, 3, 3) << std::endl;
  }

  return 0;
}