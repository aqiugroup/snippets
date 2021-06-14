#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/internal/autodiff.h>
#include <ceres/internal/numeric_diff.h>
#include <cstdlib>
#include <ctime>

// goal : use numeric diff and auto diff to check my analytic diff.
// requirement : eigen>3.3.5 ceres==1.14.0
// target function :
//            deltaQ = q_LI * q_Il1_Ir * q_Il2_Ir.conjugate() * q_LI.conjugate();
//            deltaP = q_LI * (q_Il1_Ir * (p_Ir_Il2 - p_Ir_Il1)) + p_LI - q_L1_L2_pred * p_LI;

class QuaternionCostFunctor {
public:
  QuaternionCostFunctor(const Eigen::Quaterniond &_q_LI, const Eigen::Vector3d &_p_LI,
                        const Eigen::Quaterniond &_q_L1_L2_meas, const Eigen::Vector3d &_p_L1_L2_meas) :
    q_LI_(_q_LI), p_LI_(_p_LI), q_L1_L2_meas_(_q_L1_L2_meas), p_L1_L2_meas_(_p_L1_L2_meas) {}

  template <typename T>
  bool operator()(const T *const _q_Il1_Ir, const T *const _p_Ir_Il1,
                  const T *const _q_Il2_Ir, const T *const _p_Ir_Il2,
                  T *_e) const {
    const Eigen::Quaternion<T> q_Il1_Ir(_q_Il1_Ir);
    const Eigen::Quaternion<T> q_Il2_Ir(_q_Il2_Ir);
    const Eigen::Matrix<T, 3, 1> p_Ir_Il1(_p_Ir_Il1);
    const Eigen::Matrix<T, 3, 1> p_Ir_Il2(_p_Ir_Il2);

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
    const Eigen::Matrix<T, 3, 1> p_L1_L2_pred = q_LI * (q_Il1_Ir * (p_Ir_Il2 - p_Ir_Il1)) + p_LI - q_L1_L2_pred * p_LI;

    const Eigen::Quaternion<T> e_q = q_L1_L2_pred * q_L1_L2_meas.inverse();
    const Eigen::AngleAxis<T> e_aa(e_q);
    Eigen::Matrix<T, 3, 1> e_q_delta = e_aa.axis() * e_aa.angle();
    Eigen::Map<Eigen::Matrix<T, 6, 1>> e(_e);
    // e.block<0, 0, 3, 1> = e_q_delta;
    e(0) = e_q_delta(0);
    e(1) = e_q_delta(1);
    e(2) = e_q_delta(2);

    Eigen::Matrix<T, 3, 1> e_p = p_L1_L2_pred - p_L1_L2_meas;
    // e.block<3, 0, 3, 1> = e_p;
    e(3) = e_p(0);
    e(4) = e_p(1);
    e(5) = e_p(2);

    return true;
  }

  void evaluateAnalytically(const double *const _q_Il1_Ir, const double *const _p_Ir_Il1,
                            const double *const _q_Il2_Ir, const double *const _p_Ir_Il2,
                            double *_e,
                            double **_jacobians) const {
    const Eigen::Quaternion<double> q_Il1_Ir(_q_Il1_Ir);
    const Eigen::Quaternion<double> q_Il2_Ir(_q_Il2_Ir);
    const Eigen::Matrix<double, 3, 1> p_Ir_Il1(_p_Ir_Il1);
    const Eigen::Matrix<double, 3, 1> p_Ir_Il2(_p_Ir_Il2);

    const Eigen::Quaterniond q_LI = q_LI_;
    const Eigen::Vector3d p_LI = p_LI_;

    const Eigen::Quaterniond q_L1_L2_pred = q_LI * q_Il1_Ir * q_Il2_Ir.conjugate() * q_LI.conjugate();
    Eigen::Vector3d p_L1_L2_pred = q_LI * (q_Il1_Ir * (p_Ir_Il2 - p_Ir_Il1)) + p_LI - q_L1_L2_pred * p_LI;

    const Eigen::Quaternion<double> e_q = q_L1_L2_pred * q_L1_L2_meas_.inverse();
    const Eigen::AngleAxis<double> e_aa(e_q);
    Eigen::Matrix<double, 3, 1> e_q_delta = e_aa.axis() * e_aa.angle();
    Eigen::Map<Eigen::Matrix<double, 6, 1>> e(_e);
    // e.block<0, 0, 3, 1> = e_q_delta;
    e(0) = e_q_delta(0);
    e(1) = e_q_delta(1);
    e(2) = e_q_delta(2);

    const Eigen::Matrix<double, 3, 1> e_p = p_L1_L2_pred-p_L1_L2_meas_;
    // e.block<3, 0, 3, 1> = e_p;
    e(3) = e_p(0);
    e(4) = e_p(1);
    e(5) = e_p(2);

    if (_jacobians != nullptr) {
      if (_jacobians[0] != nullptr) { // deltaQ to q_Il1_Ir
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
            _jacobians[0]);
        J.setZero();
        J.block(0, 0, 3, 3) = q_LI.toRotationMatrix();
      }
      if (_jacobians[0]+12 != nullptr) { // deltaP to q_Il1_Ir
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
            _jacobians[0]+12);
        J.setZero();
        J.block(0, 0, 3, 3) = q_LI.matrix() * (-skew(q_Il1_Ir.matrix() * (p_Ir_Il2 - p_Ir_Il1))) +
                              skew(q_L1_L2_pred.matrix() * p_LI.matrix()) * q_LI.matrix();
      }


      if (_jacobians[1]+9 != nullptr) { // deltaP to p_Ir_Il1
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(
            _jacobians[1]+9);
        J.setZero();
        J.block(0, 0, 3, 3) = -(q_LI * q_Il1_Ir).matrix();
      }


      if (_jacobians[2] != nullptr) { // deltaQ to q_Il2_Ir
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
            _jacobians[2]);
        J.setZero();
        J.block(0, 0, 3, 3) = -(q_LI * q_Il1_Ir * q_Il2_Ir.conjugate()).toRotationMatrix();
      }
      if (_jacobians[2]+12 != nullptr) { // deltaP to q_Il2_Ir
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(
            _jacobians[2]+12);
        J.setZero();
        J.block(0, 0, 3, 3) = -skew(q_L1_L2_pred.matrix() * p_LI) *
                              (q_LI * q_Il1_Ir * q_Il2_Ir.conjugate()).matrix();
      }


      if (_jacobians[3]+9 != nullptr) { // deltaP to p_Ir_Il2
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(
            _jacobians[3]+9);
        J.setZero();
        J.block(0, 0, 3, 3) = (q_LI * q_Il1_Ir).matrix();
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

  // 0 -v3 v2
  // v3 0 -v1
  // -v2 v1 0
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
  std::cout << "q_LI:\n " << q_LI.matrix() << std::endl;

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

  constexpr int RESIDUAL_NUM = 6; // delta_q : 3 + delta_p : 3
  double residuals[RESIDUAL_NUM];
  double *parameters[4] = {q_Il1_Ir.coeffs().data(), p_Ir_Il1.data(),
                           q_Il2_Ir.coeffs().data(), p_Ir_Il2.data()};
  double **jacobians = new double *[4];
  jacobians[0] = new double[24]; // 0-11: deltaQ to q_Il1_Ir 12-23: deltaP to q_Il1_Ir
  jacobians[1] = new double[18]; // 0-08: deltaQ to p_Ir_Il1 09-17: deltaP to p_Ir_Il1
  jacobians[2] = new double[24]; // 0-11: deltaQ to q_Il2_Ir 12-23: deltaP to q_Il2_Ir
  jacobians[3] = new double[18]; // 0-08: deltaQ to p_Ir_Il2 09-17: deltaP to p_Ir_Il2

  {
    std::cout << "-----------------autodiff-----------------------\n";
    ceres::internal::AutoDiff<QuaternionCostFunctor, double, 4, 3, 4, 3
                              >::Differentiate(functor, parameters,
                                                RESIDUAL_NUM, /// residual num
                                                residuals, jacobians);
    std::cout <<"rediduals ";
    for (int i = 0; i < RESIDUAL_NUM; ++i) {
      std::cout << residuals[i] << " ";
    }
    std::cout << std::endl;
    // Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> jac0(
    //     jacobians[0]);
    // std::cout << "jac0\n " << jac0 << std::endl;
    // Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jac0_0(
    //     jacobians[0]);
    // std::cout << "jac0_0\n " << jac0_0 << std::endl;
    // Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jac0_1(
    //     jacobians[0]+12);
    // std::cout << "jac0_1\n " << jac0_1 << std::endl;
    // Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> jac1(
    //     jacobians[1]);
    // std::cout << "jac[1]\n " << jac1 << std::endl;

    // deltaQ to q_Il1_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(
        jacobians[0]);
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_0;
    QuaternionCostFunctor::GlobalToLocal(parameters[0],
                                         global_to_local_0.data());
    std::cout << "autodiff jacobian_0:\n"
              << 0.5 * jacobian_0 * global_to_local_0 << std::endl;
    // deltaP to q_Il1_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(
        jacobians[0]+12);
    std::cout << "autodiff jacobian_1:\n"
              << 0.5 * jacobian_1 * global_to_local_0 << std::endl;


    // deltaP to p_Ir_Il1
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_3(
        jacobians[1]+9);
    std::cout << "autodiff jacobian_3:\n" << jacobian_3 << std::endl;


    // deltaQ to q_Il2_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_4(
        jacobians[2]);
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_1;
    QuaternionCostFunctor::GlobalToLocal(parameters[2],
                                         global_to_local_1.data());
    std::cout << "autodiff jacobian_4:\n"
              << 0.5 * jacobian_4 * global_to_local_1 << std::endl;
    // deltaP to q_Il2_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_5(
        jacobians[2]+12);
    std::cout << "autodiff jacobian_5:\n"
              << 0.5 * jacobian_5 * global_to_local_1 << std::endl;


    // deltaP to p_Ir_Il2
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_7(
        jacobians[3]+9);
    std::cout << "autodiff jacobian_7:\n" << jacobian_7 << std::endl;
  }

  {
    std::cout << "---------------numdiff-------------------------\n";
    constexpr int BLOCK_INDEX = 0;
    constexpr int BLOCK_SIZE = 4; // 扰动变量的维度
    ceres::internal::NumericDiff<
        QuaternionCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 6, 4, 3,
        4, 3, 0, 0, 0, 0, 0, 0, BLOCK_INDEX,
        BLOCK_SIZE>::EvaluateJacobianForParameterBlock(&functor, residuals,
                                              ceres::NumericDiffOptions(),
                                              6, /// residual num
                                              BLOCK_INDEX, /// block index
                                              BLOCK_SIZE, /// block size
                                              parameters, jacobians[BLOCK_INDEX]);
    // deltaQ to q_Il1_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(
        jacobians[BLOCK_INDEX]);
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_0;
    QuaternionCostFunctor::GlobalToLocal(parameters[0],
                                         global_to_local_0.data());
    std::cout << "numdiff jacobian_0:\n"
              << 0.5 * jacobian_0 * global_to_local_0 << std::endl;
    // deltaP to q_Il1_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(
        jacobians[BLOCK_INDEX]+12);
    std::cout << "numdiff jacobian_1:\n"
              << 0.5 * jacobian_1 * global_to_local_0 << std::endl;
  }

  {
    // deltaP to p_Ir_Il1
    constexpr int BLOCK_INDEX = 1;
    constexpr int BLOCK_SIZE = 3; // 扰动变量的维度
    ceres::internal::NumericDiff<
        QuaternionCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 6, 4, 3,
        4, 3, 0, 0, 0, 0, 0, 0, BLOCK_INDEX,
        BLOCK_SIZE>::EvaluateJacobianForParameterBlock(&functor, residuals,
                                              ceres::NumericDiffOptions(),
                                              6, /// residual num
                                              BLOCK_INDEX, /// block index
                                              BLOCK_SIZE, /// block size
                                              parameters, jacobians[BLOCK_INDEX]);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_3(
        jacobians[BLOCK_INDEX]+9);
    std::cout << "numdiff jacobian_3:\n"
              << jacobian_3 << std::endl;
  }

  {
    constexpr int BLOCK_INDEX = 2;
    constexpr int BLOCK_SIZE = 4; // 扰动变量的维度
    ceres::internal::NumericDiff<
        QuaternionCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 6, 4, 3,
        4, 3, 0, 0, 0, 0, 0, 0, BLOCK_INDEX,
        BLOCK_SIZE>::EvaluateJacobianForParameterBlock(&functor, residuals,
                                              ceres::NumericDiffOptions(),
                                              6, /// residual num
                                              BLOCK_INDEX, /// block index
                                              BLOCK_SIZE, /// block size
                                              parameters, jacobians[BLOCK_INDEX]);
    // deltaQ to q_Il2_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_4(
        jacobians[BLOCK_INDEX]);
    Eigen::Matrix<double, 4, 3, Eigen::RowMajor> global_to_local_1;
    QuaternionCostFunctor::GlobalToLocal(parameters[2],
                                         global_to_local_1.data());
    std::cout << "numdiff jacobian_4:\n"
              << 0.5 * jacobian_4 * global_to_local_1 << std::endl;
    // deltaP to q_Il2_Ir
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_5(
        jacobians[BLOCK_INDEX]+12);
    std::cout << "numdiff jacobian_5:\n"
              << 0.5 * jacobian_5 * global_to_local_1 << std::endl;
  }

  {
    // deltaP to p_Ir_Il2
    constexpr int BLOCK_INDEX = 3;
    constexpr int BLOCK_SIZE = 3; // 扰动变量的维度
    ceres::internal::NumericDiff<
        QuaternionCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 6, 4, 3,
        4, 3, 0, 0, 0, 0, 0, 0, BLOCK_INDEX,
        BLOCK_SIZE>::EvaluateJacobianForParameterBlock(&functor, residuals,
                                              ceres::NumericDiffOptions(),
                                              6, /// residual num
                                              BLOCK_INDEX, /// block index
                                              BLOCK_SIZE, /// block size
                                              parameters, jacobians[BLOCK_INDEX]);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_7(
        jacobians[BLOCK_INDEX]+9);
    std::cout << "numdiff jacobian_7:\n"
              << jacobian_7 << std::endl;
  }

  {
    std::cout << "------------analytic----------------------------\n";
    functor.evaluateAnalytically(parameters[0], parameters[1], parameters[2], parameters[3],
                                 residuals, jacobians);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_0(
        jacobians[0]);
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_1(
        jacobians[0]+12);


    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_3(
        jacobians[1]+9);


    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_4(
        jacobians[2]);
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_5(
        jacobians[2]+12);


    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_7(
        jacobians[3]+9);

    std::cout << "analytic jacobian_0:\n"
              << jacobian_0.block(0, 0, 3, 3) << std::endl;
    std::cout << "analytic jacobian_1:\n"
              << jacobian_1.block(0, 0, 3, 3) << std::endl;
    std::cout << "analytic jacobian_3:\n"
              << jacobian_3.block(0, 0, 3, 3) << std::endl;
    std::cout << "analytic jacobian_4:\n"
              << jacobian_4.block(0, 0, 3, 3) << std::endl;
    std::cout << "analytic jacobian_5:\n"
              << jacobian_5.block(0, 0, 3, 3) << std::endl;
    std::cout << "analytic jacobian_7:\n"
              << jacobian_7.block(0, 0, 3, 3) << std::endl;
  }

  return 0;
}
