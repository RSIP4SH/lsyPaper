#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
	Mat img = imread("001.png");
	imshow("�����ͼƬ", img);
	waitKey(0);
}