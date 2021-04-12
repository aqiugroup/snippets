//
// Created by aqiu on 2019-10-20.
//

// #include <assert.h>
// #include <time.h>
#include "MyUtils.h"

namespace MyUtils
{
    void TimeCost::StartMS()
    {
        m_startTime = std::chrono::steady_clock::now();
    }

    double TimeCost::EndMS()
    {
        m_endTime = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(m_endTime - m_startTime).count();
    }



}// namespace MyUtils
