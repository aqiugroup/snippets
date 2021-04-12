#include "SolveLinearEquation.h"
#include "MyUtils.h"

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace MyUtils;


#define MATRIX_SIZE 100
void SolveLinearEquation()
{
    // 解方程
    // 我们求解 A * x = b 这个方程
    // 直接求逆自然是最直接的，但是求逆运算量大
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > A1;
    A1 = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > b1;
    b1 = Eigen::MatrixXd::Random( MATRIX_SIZE, 1 );
    TimeCost time_stt; // 计时

    // 1 直接求逆
    time_stt.StartMS();
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = A1.inverse() * b1;
    cout << "time use in normal inverse is " << time_stt.EndMS() << "ms" << endl;
    // cout << x << endl;
    // 2.1 QR分解colPivHouseholderQr()
    // https://www.cnblogs.com/leexiaoming/p/7224781.html
    time_stt.StartMS();
    x = A1.colPivHouseholderQr().solve(b1);
    cout << "time use in Qr decomposition is " << time_stt.EndMS() << "ms" << endl;
    // cout << x << endl;
    // 2.2 QR分解fullPivHouseholderQr()
    time_stt.StartMS();
    x = A1.fullPivHouseholderQr().solve(b1);
    cout << "time use in Qr decomposition is " << time_stt.EndMS() << "ms" << endl;
    // cout << x << endl;
    // 3.1 llt分解 要求矩阵A正定
    // 正规方程法。所谓正规方程法思想是若要求解Ax=b，等价于方程两边同乘A的转置：ATAx=ATb，这样系数矩阵可化为方阵。
    time_stt.StartMS();
    x = A1.llt().solve(b1);
    cout <<"time use in llt decomposition is " << time_stt.EndMS() <<"ms" << endl;
    // cout <<x<<endl;
    // 3.2 ldlt分解  要求矩阵A正或负半定
    //正规方程法。所谓正规方程法思想是若要求解Ax=b，等价于方程两边同乘A的转置：ATAx=ATb，这样系数矩阵可化为方阵。
    time_stt.StartMS();
    x = (A1.transpose() * A1).ldlt().solve(A1.transpose() * b1);
    cout <<"time use in ldlt decomposition is " <<time_stt.EndMS() <<"ms" << endl;
    // cout <<x<<endl;
    // 4.1 lu分解 partialPivLu()
    time_stt.StartMS();
    x = (A1.transpose() * A1).partialPivLu().solve(A1.transpose() * b1);
    cout << "time use in lu decomposition is " << time_stt.EndMS() << "ms" << endl;
    // cout << x << endl;
    //4.2 lu分解（fullPivLu()
    time_stt.StartMS();
    x = A1.fullPivLu().solve(b1);
    cout << "time use in lu decomposition is " << time_stt.EndMS() << "ms" << endl;
    // cout << x << endl;

    auto A2 = A1 * A1.adjoint();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd > lu(A2);
    time_stt.StartMS();
    lu.compute(A2);
    cout << "decomposition time is " << time_stt.EndMS() << "ms" << endl;
    // ----------  结束 ----------
}
