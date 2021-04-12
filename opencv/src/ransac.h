#pragma once

#include <cmath>
#include <vector>

using namespace std;

class Point2D
{
public:
    float x, y;

public:
    Point2D(float x_ = 0, float y_ = 0) :
        x(x_), y(y_) {}

    bool operator == (const Point2D &p)
    {
        return fabs(x - p.x) < 1e-3 && fabs(y - p.y) < 1e-3;
    }
};

vector<float> Ransaclr(vector<Point2D> &pts, float outlier_prob = 0.1,
                      float accept_prob = 1e-3, float threshold = 10.0);