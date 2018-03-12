#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

class ImagePreprocessBase
{
public:
	ImagePreprocessBase();
	virtual ~ImagePreprocessBase();
private:
};

class ImagePreprocess : public ImagePreprocessBase
{
public:
	ImagePreprocess();
	virtual ~ImagePreprocess();
private:
};