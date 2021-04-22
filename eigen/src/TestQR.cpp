#include "TestQR.h"
#include "MyUtils.h"
#include <Eigen/QR>

#include <iostream>
#include <cmath>

using namespace std;
using namespace MyUtils;
using namespace Eigen;

void HouseholderQRinplace(Ref<MXX> &qr, unsigned int blockSize) {
    VXX hCoef, tmp;
    Eigen::Index row = qr.rows();
    Eigen::Index col = qr.cols();
    Eigen::Index size = std::min(row, col);
    hCoef.resize(size);
    tmp.resize(col);
    Eigen::internal::householder_qr_inplace_blocked<Ref<MXX>, VXX>::run(qr, hCoef, blockSize, tmp.data());
}

void FactorUpdateQR()
{
    TimeCost time_stt;time_stt.StartMS();
    const int beginIdx = 1;
    const int nObs = 4; // 3state+1obeservation(4x4)
    const int nStates = 3; // states nums
    MXX resMat = MXX::Random(nObs, nObs); // last col was residual(4x4)
    resMat.topLeftCorner(nObs, nObs).triangularView<StrictlyLower>().setZero();
    resMat(3,1)=0.1; // for test
    cout << "origin \n" << resMat << endl;

    VXI orderIndices(nObs-beginIdx); // exchange first two cols state(last was residuan)
    orderIndices(0)=1;
    orderIndices(1)=0;
    orderIndices(2)=2;
    cout << "orderIndices " << orderIndices.transpose() << endl;
    PermutationMatrix<Dynamic, Dynamic> per(orderIndices);
    cout << "per \n" << per.toDenseMatrix() << endl;
    Ref<MXX> qrMat = resMat.block(beginIdx, beginIdx, nObs-beginIdx, nStates+1-beginIdx);//mid 2x2
    cout << "before permutation \n" << qrMat << endl;
    qrMat.applyOnTheLeft(per);
    cout << "after permutaion qrMat \n" << qrMat << endl;
    qrMat.applyOnTheLeft(per);
    cout << "after permutaion qrMat \n" << qrMat << endl;

    int remainSize = qrMat.cols()-1;
    HouseholderQRinplace(qrMat, 12);
    Ref<MXX> newRes = resMat.block(0, nStates, nStates, 1);
    cout << "after QR qrMat \n" << qrMat << endl;
    cout << "topLeftCorner \n" << qrMat.topLeftCorner(remainSize, remainSize) << endl;

    // TODO remainSize below should clear, but computational
    qrMat.topLeftCorner(remainSize, remainSize).triangularView<StrictlyLower>().setZero();
    cout << "triangularView<StrictlyLower>().setZero \n" << qrMat << endl;
    cout << "newRes " << newRes.transpose() << endl;
    VXX updateDelta = resMat.topLeftCorner(nStates, nStates).triangularView<Upper>().solve(newRes);
    cout << "updateDelta " << updateDelta.transpose() << endl;

    cout << "decomposition time is " << time_stt.EndMS() << "ms" << endl;
}
