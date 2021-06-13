#include "virtual.h"

typedef void(*Fun)(void);  //函数指针

void TestChildNoVirtual()
{
    CTest t;
    printf("CTest 虚表地址:%p\n", *(int *)&t);
    CTestC t0;
    printf("CTestC 虚表地址:%p\n", *(int *)&t0 );
    Fun pfun = (Fun) * ((int *) * (int *)(&t0)); //vitural f();
    printf("f():%p\n", pfun);
    pfun();
}