#ifndef MYUTILS_H
#define MYUTILS_H

#include <chrono>

namespace MyUtils
{
    /*----------------- TimeCost start -----------------*/
    class TimeCost
    {
    public:
        void StartMS();
        double EndMS();

    private:
        std::chrono::steady_clock::time_point m_startTime;
        std::chrono::steady_clock::time_point m_endTime;
    };
    /*----------------- TimeCost end -----------------*/
}

#endif