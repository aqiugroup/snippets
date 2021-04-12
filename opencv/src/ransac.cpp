
#include "ransac.h"
#include <ctime>
#include <iostream>
#include <limits>
#include <tuple>

vector<float> Ransaclr(vector<Point2D> &pts, float outlier_prob,
                       float accept_prob, float threshold)
{
    int n = pts.size();
    float sample_fail_prob = 1 - (1 - outlier_prob) * (1 - outlier_prob); // 1 - t^N
    int K = log(accept_prob) / log(sample_fail_prob); // accept_prob = 1 - P, sample_fail_prob = 1 - t^N

    float a_res, b_res, c_res;
    float min_error = numeric_limits<float>::max();
    //cout << "K = " << K << endl;
    for (int k = 0; k < K; ++k) {
        Point2D p1, p2;
        while (p1 == p2) { // 1 随机选取两个不相等的点
            p1 = pts[rand() % n];    // n maybe greater than 65535
            p2 = pts[rand() % n];
        }
        //cout << "p1 = " << p1.x << " " << p1.y << endl;
        //cout << "p2 = " << p2.x << " " << p2.y << endl;
        float a, b, c; // 2 估计参数a、b、c
        a = p1.y - p2.y;
        b = p2.x - p1.x;
        c = p1.x * p2.y - p1.y * p2.x;
        float t = sqrt(a * a + b * b);
        a /= t;
        b /= t;
        c /= t;
        //cout << a << " " << b << " " << c << endl;
        float error = 0.0;
        int inliers = 0;
        for (int i = 0; i < n; ++i) { // 3 在当前随机选取的两个点的假设下，计算误差， 统计内点数
            float dis = fabs(a * pts[i].x + b * pts[i].y + c);
            if (dis < threshold) {
                ++inliers;
                error += dis;
            }
        }
        //cout << "inliers = " << inliers << endl;
        //cout << "error = " << error << endl;
        if (static_cast<float>(inliers) / static_cast<float>(n) > 0.7) { // 4 内点率大于0.7 即认为算是一个好的假设
            if (error < min_error) {
                min_error = error;
                a_res = a;
                b_res = b;
                c_res = c;
            }
        }
    }
    return vector<float> { a_res, b_res, c_res };
}
