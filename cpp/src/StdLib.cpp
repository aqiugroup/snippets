#include "StdLib.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace MyUtils;

void test_nth_element()
{
    std::vector<int> v{5, 6, 4, 3, 2, 6, 7, 9, 1, 3};
    int len = v.size();
    std::nth_element(v.begin(), v.begin() + len / 2, v.end());
    for(int i = 0; i < len; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "The median is " << v[len / 2] << '\n';
    std::nth_element(v.begin(), v.begin() + 1, v.end(), std::greater<int>());
    for(int i = 0; i < len; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "The second largest element is " << v[1] << '\n';
}
