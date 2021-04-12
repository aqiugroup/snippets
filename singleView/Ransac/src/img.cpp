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
    if(image.isContinuous()) { //�ж��Ƿ����ڴ�������
        cout << "image storage is continuous " << endl;
    }
    //
    // ֪ʶ��1 �� ��ǳ����
    // cv::Mat imageCopy(image); // ǳ����
    // cv::Mat imageCopy=image; // ǳ����
    cv::Mat imageCopy; // ���
    image.copyTo(imageCopy);
    cv::Mat imageCopyPtr = image.clone(); // ���
    // cv::namedWindow("color-ori", CV_WINDOW_AUTOSIZE); // Create a window for display.
    imshow("color-ori", image);
    //
    // ֪ʶ��2 �� ͼ��ԭʼָ����ʣ�����ͨ����ȡֵuchar,Vec3b,Vec4b
    int rows = image.rows;
    int cols = image.cols;
    TimeCost t1;
    t1.StartMS();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //at<����>(i,j)���в��������ڻҶ�ͼ
            image.at<Vec4b>(i, j)[0] = 255;//������ɫͨ�����в���
            //img.at<Vec4b>(i, j)[1] = 255;//������ɫͨ�����в���
            //img.at<Vec4b>(i, j)[2] = 255;//���ں�ɫͨ�����в���
        }
    }
    float time1 = t1.EndMS();
    imshow("color-result", image);
    cvtColor(imageCopy, imageCopy, CV_RGB2GRAY);//תΪ�Ҷ�ͼ
    imshow("gray_ori", imageCopy);
    //
    // ֪ʶ��2 �� ͼ��ԭʼָ����ʣ�����ͨ����ȡֵuchar
    rows = imageCopyPtr.rows;
    cols = imageCopy.cols;
    TimeCost t2;
    t2.StartMS();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //at<����>(i,j)���в��������ڻҶ�ͼ
            imageCopy.at<uchar>(i, j) = i + j;
        }
    }
    float time2 = t2.EndMS();
    imshow("gray_result", imageCopy);
    imshow("ptr_ori", imageCopyPtr);
    //
    // ֪ʶ��3 �� ͼ������ָ����ʣ�����ͨ����ȡ����
    TimeCost t3;
    t3.StartMS();
    rows = imageCopyPtr.rows;
    cols = imageCopyPtr.cols * imageCopyPtr.channels();
    for(int i = 0; i < rows; i++) {
        //��ȡ�洢ͼ���ڴ�ĵ�i�е�ָ��
        uchar *pointer = imageCopyPtr.ptr<uchar>(i);
        for(int j = 0; j < cols; j += 4) {
            //pointer[j] = 255;//����ɫͨ�����в���
            //pointer[j+1] = 255;//����ɫͨ�����в���
            pointer[j + 2] = 255; //�Ժ�ɫͨ�����в���
        }
    }
    float time3 = t3.EndMS();
    //
    // ֪ʶ��4 �� ͼ������ָ����ʣ�Ч�ʿ�Щ
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


vector<float> getHist1(Mat &image)//RGBɫ�ʿռ�ģ�4��4��4��ģ�͡�
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
    for (int i = 0; i < length; i++) { //��һ��
        a[i] = a[i] / sum;
        b.push_back(a[i]);
    }
    return b;
}


vector<float> getHist2(Mat &image)//RGBɫ�ʿռ�ģ�4��4��4��ģ�͡�
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
    for (int i = 0; i < div; i++) { //��һ��
        a[i] = a[i] / sum;
        bin_hist.push_back(a[i]);
    }
    return bin_hist;
}


// ����ͼ
void GetGrayIntegralImage(unsigned char *Src, int *Integral, int Width, int Height)
{
    memset(Integral, 0, (Width + 1) * sizeof(int));    // ��һ�ж�Ϊ0
    for (int Y = 0; Y < Height; Y++) {
        unsigned char *LineSrc = Src + Y * Width;      // ԭͼ����ÿһ�е���ʼλ��
        int *LinePre = Integral + Y * (Width + 1) + 1; // ��һ��λ��
        int *LineCur = Integral + (Y + 1) * (Width + 1) + 1; // ��ǰλ�ã�ע��ÿ�еĵ�һ�е�ֵ��Ϊ0
        LineCur[-1] = 0;                                     // ��һ�е�ֵΪ0
        //��һ����Ҳ��ͷһ�μ���������±����Ϊ��������ʾ�ڴ��е���һλ��
        for (int X = 0, Sum = 0; X < Width; X++) {
            Sum += LineSrc[X];                    // �з����ۼ�
            LineCur[X] = LinePre[X] + Sum;        // ���»���ͼ
        }
    }
}





