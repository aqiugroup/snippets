#include "MyUtils.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <stdio.h>
using namespace std;

void DetectBall(cv::Mat &img) {
	//  HSV(Hue) 1D Gaussian
	float mu = 39.5989;
	float sig = 3.9681;
	// % 阈值根据参数自动算出，取2σ的概率作为阈值
	float thre = exp(-pow(2*sig,2)/(2*sig*sig))/(sqrt(2*M_PI)*sig);

	// % HSV方法
	cv::Mat img_hsv;
	cvtColor(img, img_hsv, CV_BGR2HSV);
	imshow("img_hsv", img_hsv);
	cv::moveWindow("img_hsv", img.rows*img.channels(), 0);

	// cv::Mat channel_h = img_hsv(:,:,1) * 256; // 前面求解参数的时候乘了256，这里保持统一，也乘以256
	// // % 计算每个像素的概率
	// cv::Mat prob_img = exp(-pow(channel_h-mu,2)/(2*sig*sig))/(sqrt(2*pi)*sig);
	// cv::Mat prob_mask = prob_img>thre;  // 阈值操作，初步确定掩膜

	// // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// bw_biggest = false(size(prob_mask));
	// CC = bwconncomp(prob_mask);    //% 寻找连通域像素
	// numPixels = cellfun(@numel,CC.PixelIdxList);    //% 获取每个连通域包含的像素个数
	// [biggest,idx] = max(numPixels); //% 找到包含像素数最多的连通域的像素个数以及其所对应的索引idx(后面会用到)
	// bw_biggest(CC.PixelIdxList{idx}) = true;    //% 根据索引将最大连通域内的所有像素设为true

	// // % Compute the location of the ball center
	// S = regionprops(CC,'Centroid'); //% 计算出所有连通域的重心
	// loc = S(idx).Centroid;  //% 根据上面获得的索引，获取我们想要的重心坐标
	// segI = bw_biggest;
}

int main( int argc, char** argv )
{
	cout << " trackBall " << endl;

	string imagepath("./../interestingProject/trackBall/train");
    cv::Mat image;
	char buffer [50];
	for (int i = 1; i < 19; ++i)
	{
  		int n = sprintf (buffer, "%s/%03d.png", imagepath.c_str(), i);
  		printf ("[%s] is a string %d chars long\n", buffer, n);
  		// string st1 = buffer;
		// image = cv::imread(st1, CV_LOAD_IMAGE_UNCHANGED);   // Read the file
		image = cv::imread(buffer, CV_LOAD_IMAGE_UNCHANGED);   // Read the file
	    if (!image.data) {
	        std::cout << "Could not open the image!" << std::endl;
	        return -1;
	    }
		cout << "image row " << image.rows << endl;
		cout << "iamge col " << image.cols << endl;
		cout << "image channel " << image.channels() << endl;

	    imshow("origin", image);
		cv::moveWindow("origin", 0, 0);
	    DetectBall(image);

	    cv::waitKey(1000);
	}


	// for k=1:19
	//     % Load image
	//     I = imread(sprintf('%s/%03d.png',imagepath,k));

	//     % Implement your own detectBall function
	//     [segI, loc] = detectBall(I);
	//     set(gca,'position',[10,10,10,10])
	//     set(gcf,'position',[300, 300,550, 300])
	//     subplot(1,2,1);
	//     title(k);
	//     imshow(segI); hold on; 
	//     plot(loc(1), loc(2), '+b','MarkerSize',7); 
	//     subplot(1,2,2);
	//     imshow(I);

	//     disp('Press any key to continue. (Ctrl+c to exit)')
	//     pause
	// end


    return 0;
}
