#pragma once
#include "MyUtils.h"

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace MyUtils;

class HistogramND
{
private:
    Mat image;//源图像
    int hisSize[1], hisWidth, hisHeight;//直方图的大小,宽度和高度
    float range[2];//直方图取值范围
    const float *ranges;
    Mat channelsRGB[3];//分离的BGR通道
    MatND outputRGB[3];//输出直方图分量
public:
    HistogramND()
    {
        hisSize[0] = 256;
        hisWidth = 400;
        hisHeight = 400;
        range[0] = 0.0;
        range[1] = 255.0;
        ranges = &range[0];
    }

    //导入图片
    bool loadImage(String path)
    {
        image = imread(path);
        if (!image.data) {
            return false;
        }
        return true;
    }

    //分离通道
    void splitChannels()
    {
        split(image, channelsRGB);
    }

    //计算直方图
    void getHistogram()
    {
        calcHist(&channelsRGB[0], 1, 0, Mat(), outputRGB[0], 1, hisSize, &ranges);
        calcHist(&channelsRGB[1], 1, 0, Mat(), outputRGB[1], 1, hisSize, &ranges);
        calcHist(&channelsRGB[2], 1, 0, Mat(), outputRGB[2], 1, hisSize, &ranges);
        //输出各个bin的值
        for (int i = 0; i < hisSize[0]; ++i) {
            cout << i << "   B:" << outputRGB[0].at<float>(i);
            cout << "   G:" << outputRGB[1].at<float>(i);
            cout << "   R:" << outputRGB[2].at<float>(i) << endl;
        }
    }

    //显示直方图
    void displayHisttogram()
    {
        Mat rgbHist[3];
        for (int i = 0; i < 3; i++) {
            rgbHist[i] = Mat(hisWidth, hisHeight, CV_8UC3, Scalar::all(0));
        }
        normalize(outputRGB[0], outputRGB[0], 0, hisWidth - 20, NORM_MINMAX);
        normalize(outputRGB[1], outputRGB[1], 0, hisWidth - 20, NORM_MINMAX);
        normalize(outputRGB[2], outputRGB[2], 0, hisWidth - 20, NORM_MINMAX);
        for (int i = 0; i < hisSize[0]; i++) {
            int val = saturate_cast<int>(outputRGB[0].at<float>(i));
            rectangle(rgbHist[0], Point(i * 2 + 10, rgbHist[0].rows), Point((i + 1) * 2 + 10, rgbHist[0].rows - val), Scalar(0, 0, 255), 1, 8);
            val = saturate_cast<int>(outputRGB[1].at<float>(i));
            rectangle(rgbHist[1], Point(i * 2 + 10, rgbHist[1].rows), Point((i + 1) * 2 + 10, rgbHist[1].rows - val), Scalar(0, 255, 0), 1, 8);
            val = saturate_cast<int>(outputRGB[2].at<float>(i));
            rectangle(rgbHist[2], Point(i * 2 + 10, rgbHist[2].rows), Point((i + 1) * 2 + 10, rgbHist[2].rows - val), Scalar(255, 0, 0), 1, 8);
        }
        cv::imshow("R", rgbHist[0]);
        imshow("G", rgbHist[1]);
        imshow("B", rgbHist[2]);
        imshow("image", image);
    }
};


void MyCalHist();

vector<float> getHist1(Mat &image);//RGB色彩空间的（4，4，4）模型。
vector<float> getHist2(Mat &image);//RGB色彩空间的（4，4，4）模型。


void ReadAndDisplayImage();
