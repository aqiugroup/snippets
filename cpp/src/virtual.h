
#include <iostream>

class CTest
{
public:
    CTest() {};
    ~CTest() {};
    virtual void SFunc()
    {
        std::cout << "测试虚函数" << std::endl;
    };
private:
    int a;
    short b;
};

class CTestC : public CTest
{
public:
    CTestC()
    {
        m_iValueC = 0;
    };
    ~CTestC() {};

    virtual void Test()
    {
        std::cout << "测试" << std::endl;
    };
private:
    int m_iValueC;
};

void TestChildNoVirtual();