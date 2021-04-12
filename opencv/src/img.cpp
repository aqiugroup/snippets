#include "img.h"
#include "MyUtils.h"

#include <iostream>
#include <opencv2/opencv.hpp>


void ReadAndDisplayImage()
{
    cv::Mat image;
    image = cv::imread("../opencv/data/google.png", CV_LOAD_IMAGE_UNCHANGED);   // Read the file
    if (!image.data) {
        std::cout << "Could not open the image!" << std::endl;
        return;
    }
    cout << "image row " << image.rows << endl;
    cout << "iamge col " << image.cols << endl;
    cout << "image channel " << image.channels() << endl;
    if(image.isContinuous()) { //判断是否在内存中连续
        cout << "image storage is continuous " << endl;
    }
    //
    // 知识点1 ： 深浅拷贝
    // cv::Mat imageCopy(image); // 浅拷贝
    // cv::Mat imageCopy=image; // 浅拷贝
    cv::Mat imageCopy; // 深拷贝
    image.copyTo(imageCopy);
    cv::Mat imageCopyPtr = image.clone(); // 深拷贝
    // cv::namedWindow("color-ori", CV_WINDOW_AUTOSIZE); // Create a window for display.
    imshow("color-ori", image);
    //
    // 知识点2 ： 图像原始指针访问，根据通道数取值uchar,Vec3b,Vec4b
    int rows = image.rows;
    int cols = image.cols;
    TimeCost t1;
    t1.StartMS();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //at<类型>(i,j)进行操作，对于灰度图
            image.at<Vec4b>(i, j)[0] = 255;//对于蓝色通道进行操作
            //img.at<Vec4b>(i, j)[1] = 255;//对于绿色通道进行操作
            //img.at<Vec4b>(i, j)[2] = 255;//对于红色通道进行操作
        }
    }
    float time1 = t1.EndMS();
    imshow("color-result", image);
    cvtColor(imageCopy, imageCopy, CV_RGB2GRAY);//转为灰度图
    imshow("gray_ori", imageCopy);
    //
    // 知识点2 ： 图像原始指针访问，根据通道数取值uchar
    rows = imageCopyPtr.rows;
    cols = imageCopy.cols;
    TimeCost t2;
    t2.StartMS();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //at<类型>(i,j)进行操作，对于灰度图
            imageCopy.at<uchar>(i, j) = i + j;
        }
    }
    float time2 = t2.EndMS();
    imshow("gray_result", imageCopy);
    imshow("ptr_ori", imageCopyPtr);
    //
    // 知识点3 ： 图像智能指针访问，根据通道数取步长
    TimeCost t3;
    t3.StartMS();
    rows = imageCopyPtr.rows;
    cols = imageCopyPtr.cols * imageCopyPtr.channels();
    for(int i = 0; i < rows; i++) {
        //调取存储图像内存的第i行的指针
        uchar *pointer = imageCopyPtr.ptr<uchar>(i);
        for(int j = 0; j < cols; j += 4) {
            //pointer[j] = 255;//对蓝色通道进行操作
            //pointer[j+1] = 255;//对绿色通道进行操作
            pointer[j + 2] = 255; //对红色通道进行操作
        }
    }
    float time3 = t3.EndMS();
    //
    // 知识点4 ： 图像智能指针访问，效率快些
    cout << "t1 " << time1 << " t2 " << time2 << " ptr-faster t3 " << time3 << endl;
    imshow("ptr_result", imageCopyPtr);
    cv::waitKey(0);
}



void MyCalHist()
{
    string path = "../opencv/data/google.png";
    HistogramND hist;
    if (!hist.loadImage(path)) {
        cout << "Import Error!" << endl;
        return ;
    }
    hist.splitChannels();
    hist.getHistogram();
    hist.displayHisttogram();
    // getHist1(path);
    // getHist2(path);
    waitKey(0);
}


vector<float> getHist1(Mat &image)//RGB色彩空间的（4，4，4）模型。
{
    const int div = 64;
    const int bin_num = 256 / div;
    int length = bin_num * bin_num * bin_num;
    Vec3i pix;
    vector<float> b;
    float a[bin_num * bin_num * bin_num] = { 0 };
    float d[bin_num][bin_num][bin_num] = { 0 };
    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            pix = image.at<Vec3b>(r, c);
            pix.val[0] = pix.val[0] / div;
            pix.val[1] = pix.val[1] / div;
            pix.val[2] = pix.val[2] / div;
            //  image.at<Vec3b>(r, c) = pix;
            d[pix.val[0]][pix.val[1]][pix.val[2]]++;
        }
    }
    float sum = 0;
    for (int i = 0; i < bin_num; i++)
        for (int j = 0; j < bin_num; j++)
            for (int k = 0; k < bin_num; k++) {
                a[i * bin_num * bin_num + j * bin_num + k] = d[i][j][k];
                sum += d[i][j][k];
                cout << d[i][j][k] << " ";
            }
    for (int i = 0; i < length; i++) { //归一化
        a[i] = a[i] / sum;
        b.push_back(a[i]);
    }
    return b;
}


vector<float> getHist2(Mat &image)//RGB色彩空间的（4，4，4）模型。
{
    const int div = 64;
    const int bin_num = 256 / div;
    int nr = image.rows; // number of rows
    int nc = image.cols; // number of columns
    if (image.isContinuous())  {
        // then no padded pixels
        nc = nc * nr;
        nr = 1;  // it is now a 1D array
    }
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));// mask used to round the pixel value
    uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0
    int b, g, r;
    vector<float> bin_hist;
    int ord = 0;
    float a[bin_num * bin_num * bin_num] = { 0 };
    for (int j = 0; j < nr; j++) {
        const uchar *idata = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++) {
            b = ((*idata++)&mask) / div;
            g = ((*idata++)&mask) / div;
            r = ((*idata++)&mask) / div;
            ord = b * bin_num * bin_num + g * bin_num + r;
            a[ord] += 1;
        } // end of row
    }
    float sum = 0;
    // cout << "a[i]2:" << endl;
    for (int i = 0; i < div; i++) {
        sum += a[i];
        // cout << a[i] << " ";
    }
    for (int i = 0; i < div; i++) { //归一化
        a[i] = a[i] / sum;
        bin_hist.push_back(a[i]);
    }
    return bin_hist;
}


// 积分图
void GetGrayIntegralImage(unsigned char *Src, int *Integral, int Width, int Height)
{
    memset(Integral, 0, (Width + 1) * sizeof(int));    // 第一行都为0
    for (int Y = 0; Y < Height; Y++) {
        unsigned char *LineSrc = Src + Y * Width;      // 原图像中每一行的起始位置
        int *LinePre = Integral + Y * (Width + 1) + 1; // 上一行位置
        int *LineCur = Integral + (Y + 1) * (Width + 1) + 1; // 当前位置，注意每行的第一列的值都为0
        LineCur[-1] = 0;                                     // 第一列的值为0
        //这一步我也是头一次见，数组的下标可以为负数，表示内存中的上一位置
        for (int X = 0, Sum = 0; X < Width; X++) {
            Sum += LineSrc[X];                    // 行方向累加
            LineCur[X] = LinePre[X] + Sum;        // 更新积分图
        }
    }
}





