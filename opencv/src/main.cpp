#include "img.h"
// #include <cstdlib>
#include "ransac.h"
#include "kdtree.h"

int main(int argc, char **argv)
{
    // ReadAndDisplayImage();
    // MyCalHist();
    // 1 ransac
    int n = 5;
    vector<Point2D> pts{{3, 4}, {6, 8}, {9, 12}, {15, 20}, {10, -10}};
    auto params = Ransaclr(pts, 0.1, 1e-4, 10.0);
    cout << params[0] << " " << params[1] << " " << params[2] << endl;
    int NUM  = 9999999;
    int sum = 0;
    srand(time(NULL));
    for (int i = 0; i < NUM; i++) {
        double val = 0;
        while(val < 1) {
            val += (rand() / (double)RAND_MAX);
            sum++;
        }
    }
    printf("%f\n", sum / (double)NUM);
    // 2 TestKdtree
    if (argc > 1) {
        const int seed = std::stoi(argv[1]);
        TestKdtree(seed);
    } else {
        printf("!!!!please input seed for TestKdtree !!!!\n");
        // TestKdtree();
    }
    return 0;
}

