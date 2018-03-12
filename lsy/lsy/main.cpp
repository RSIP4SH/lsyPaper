
////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include<iostream>
#include <time.h>
using namespace std;
using namespace cv;
//区域生长法
//////////////////////////
IplImage* RegionGrow(int x, int y, IplImage *src, int gate) //这里的x、y指第x行y列
{
	//8邻域对应坐标数组
	int indexx[] = { -1,-1,-1,0,0,1,1,1 };
	int indexy[] = { -1,0,1,-1,1,-1,0,1 };
	int k;//循环控制变量

		  //定义指针(一维数组)以存储坐标
	int *m_RegionGrowX;
	int *m_RegionGrowY;
	int *m_RegionGrowFlag;
	//开辟空间
	m_RegionGrowX = new int[8 * src->width*src->height];
	m_RegionGrowY = new int[8 * src->width*src->height];
	m_RegionGrowFlag = new int[8 * src->width*src->height];
	for (int i = 0; i<src->height; i++)
		for (int j = 0; j<src->width; j++)
		{
			m_RegionGrowFlag[i*src->widthStep + j] = 0;

		}
	//定义的起点和终点
	int m_Start;
	int m_End;
	//赋初值
	m_Start = 0;
	m_End = 0;
	//把起始种子点坐标存入数组
	m_RegionGrowX[m_End] = x;
	m_RegionGrowY[m_End] = y;


	//当前点坐标（中心点坐标）也就是种子点
	int m_CurrX;
	int m_CurrY;
	//新的点坐标（邻域点坐标）
	int m_NewX;
	int m_NewY;
	while (m_Start<(src->width*src->height) - 1)
	{

		while (m_Start <= m_End)
		{
			//当前点坐标赋值，用以当做种子在邻域中寻找灰度接近的点
			m_CurrX = m_RegionGrowX[m_Start];
			m_CurrY = m_RegionGrowY[m_Start];
			for (k = 0; k<8; k++)
			{
				m_NewX = m_CurrX + indexx[k];
				m_NewY = m_CurrY + indexy[k];
				if ((m_NewX<src->height) && (m_NewY<src->width) && (m_NewX >= 0) && (m_NewY >= 0))  //判断邻域点在图像范围内才能进行下述操作
				{
					uchar temp = uchar((src->imageData + src->widthStep*m_CurrX)[m_CurrY]);
					uchar temp1 = uchar((src->imageData + src->widthStep*m_NewX)[m_NewY]);
					int qq;
					qq = int(temp - temp1);
					int pp;
					pp = m_NewX*src->width + m_NewY;
					int mm;
					mm = m_RegionGrowFlag[pp];
					// 生长条件：判断像素(m_NewX,m_NewY)和当前像素(m_CurrX,m_CurrY) 像素值差的绝对值<=gate则进行生长
					if ((mm == 0) && (abs(qq) <= gate))
					{
						m_End++;
						//将被生长点的坐标存入数组以便用来当做下次生长的起点（种子）
						m_RegionGrowX[m_End] = m_NewX;
						m_RegionGrowY[m_End] = m_NewY;
						m_RegionGrowFlag[m_NewX*src->width + m_NewY] = 1;//标记已读像素点
					}
				}
			}
			m_Start++;

		}
		m_End = m_Start;
		m_RegionGrowX[m_End] = int(m_End / src->width);
		m_RegionGrowY[m_End] = m_End - m_RegionGrowX[m_End] * src->width;
	}

	delete[]m_RegionGrowX;
	delete[]m_RegionGrowY;
	m_RegionGrowX = NULL;
	m_RegionGrowY = NULL;
	IplImage* dst = cvCloneImage(src);
	for (int i = 0; i<src->height; i++)
		for (int j = 0; j<src->width; j++)
		{
			//将被生长点灰度值设为0其它点设为255
			if (m_RegionGrowFlag[i*src->width + j] == 1)
				(dst->imageData + dst->widthStep*i)[j] = 0;
			else
				(dst->imageData + dst->widthStep*i)[j] = 255;
		}
	delete[]m_RegionGrowFlag;
	m_RegionGrowFlag = NULL;
	return(dst);

};


//K均值聚类
///////////////////////////////
IplImage* K_Means(IplImage* src)
{
	int i, j, k = 0, value;
	int nCuster = 3;//给定聚类数目
					//定义数组cluster用以标志每个样本对应的类别，取值范围0,1,2...nCuster-1;
	CvMat* clusters = cvCreateMat(src->height*src->width, 1, CV_32SC1);//Opencv内部函数cvKMeans2要求label数组必须是CV_32SC1型
	CvMat* samples = cvCreateMat(src->height*src->width, 1, CV_32FC2);//要求sampels数组必须是CV_32FC2型
	IplImage* dst = cvCreateImage(cvGetSize(src), 8, 1);
	for (i = 0; i<src->width; i++)
		for (j = 0; j<src->height; j++)
		{
			CvScalar s;
			//获取图像各个像素点的三通道值(BGR)
			s.val[0] = (float)cvGet2D(src, j, i).val[0];
			s.val[1] = (float)cvGet2D(src, j, i).val[1];
			s.val[2] = (float)cvGet2D(src, j, i).val[2];
			cvSet2D(samples, k++, 0, s);

		}

	cvKMeans2(samples, nCuster, clusters, cvTermCriteria(CV_TERMCRIT_ITER, 100, 1.0));
	//绘制聚类后的图像
	k = 0;
	float step = 255 / (nCuster - 1);
	for (i = 0; i<src->width; i++)
	{
		for (j = 0; j<src->height; j++)
		{
			value = clusters->data.i[k++];
			CvScalar s;
			s.val[0] = 255 - value*step;
			cvSet2D(dst, j, i, s);

		}
	}
	return(dst);
};


//模糊C聚类算法准备函数
///////////////////////////////
double** Standardize(double **data, int row, int col)
{
	int i, j;
	double *a = new double[col];//矩阵每列最大值
	double *b = new double[col];//矩阵每列最小值
	double *c = new double[row];//用以暂时存储矩阵某一列元素
	for (i = 0; i<col; i++)
	{
		//取出数据矩阵的各列元素
		for (j = 0; j<row; j++)
		{
			c[j] = data[j][i];
		}
		a[i] = c[0];
		b[i] = c[0];
		for (j = 0; j<row; j++)
		{
			//列最大值
			if (c[j]>a[i])
				a[i] = c[j];
			//列最小值
			if (c[j]<b[i])
				b[i] = c[j];
		}
	}
	for (i = 0; i<row; i++)
	{
		for (j = 0; j<col; j++)
		{
			data[i][j] = (data[i][j] - b[j]) / (a[j] - b[j]);
		}
	}
	cout << "完成数据极差标准化处理>>>>>>>>>\n";
	delete[]a;
	delete[]b;
	delete[]c;

	return(data);

}
//生成样本隶属度矩阵
void Initialize(double **u, int k, int row)//k为聚类数
{
	int i, j;
	//初始化样本隶属度矩阵
	srand((unsigned)time(0));
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			u[i][j] = (double)rand() / RAND_MAX;//隶属度取值范围设为0~1
		}//rand()函数返回0~RANDN_MAX之间的一个伪随机数
	}
	//隶属度数据归一化
	double *sum = new double[row];//隶属度矩阵每列的和
	for (j = 0; j<row; j++)//这里用row表示列数不是拼写错误而是因为隶属度矩阵的列数等于data矩阵的行数，row是data矩阵的行
	{
		double dj = 0;
		for (i = 0; i<k; i++)
		{
			dj += u[i][j];
		}
		sum[j] = dj;//隶属度矩阵各列之和

	}
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			u[i][j] /= sum[j];//归一化
		}
	}

	cout << "样本隶属度矩阵生成>>>>>>>>>>>>>" << endl;
	delete[]sum;
}
//迭代函数
double Update(double **u, double **data, double **center, int row, int col, int k, int m, double **U, double **dis, double *a, double *b)
{
	int i, j, t;
	/*double **U=new double *[k];
	for(j=0;j<k;j++)
	{
	U[j]=new double[row];
	}*/
	double si = 0;//center(i,j) 的分子
	double sj = 0;//center(i,j) 的分母
				  //根据隶属度矩阵计算聚类中心，参见论文3.4式
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			U[i][j] = pow(u[i][j], m);//m为模糊指数，越高越模糊，越小越接近K均值聚类

		}

	}
	for (j = 0; j<col; j++)
	{
		for (i = 0; i<k; i++)
		{
			for (t = 0; t<row; t++)
			{
				si += U[i][t] * data[t][j];
				sj += U[i][t];
			}
			center[i][j] = si / sj;
		}
	}
	//计算各个聚类中心i分别到所有点j的距离矩阵dis(i,j)
	/*double *a=new double[col];
	double *b=new double[col];
	double **dis=new double *[k];//聚类中心与样本之间的距离矩阵

	for(i=0;i<k;i++)
	{
	dis[i]=new double[row];
	}*/
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<col; j++)
			a[j] = center[i][j];//暂存一个类中心
		for (j = 0; j<row; j++)
		{
			for (t = 0; t<col; t++)
				b[t] = data[j][t];//暂存一个样本
			double d = 0;
			//计算聚类中心与样本之间的距离
			for (t = 0; t<col; t++)
			{
				d += (a[t] - b[t])*(a[t] - b[t]);//d为一个中心与一个样本的欧几里得距离的平方
			}
			dis[i][j] = sqrt(d);//欧几里得距离
		}
	}
	//隶属度矩阵的更新，参见论文的3.5式
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			double temp = 0;
			for (t = 0; t<k; t++)
			{
				//模糊指数为m
				temp += pow(dis[i][j] / dis[t][j], 2 / (m - 1));
			}
			u[i][j] = 1 / temp;
		}
	}
	//根据FCM的价值函数（目标函数）计算聚类有效性评价参数参见论文中3.2式
	double func1 = 0;
	for (i = 0; i<k; i++)
	{
		double func2 = 0;
		for (j = 0; j<row; j++)
		{
			func2 += U[i][j] * (dis[i][j] * dis[i][j]);
		}
		func1 += func2;
	}
	double obj_fun = func1;
	//double obj_fun=1/(1+func1);
	return(obj_fun);
	/*
	//内存释放
	for(j=0;j<k;j++)
	{
	delete[]U[j];
	}
	delete[]U;
	delete[]a;
	delete[]b;
	for(i=0;i<k;i++)
	delete[]dis[i];
	delete[]dis;
	*/
}
//模糊C均值聚类算法（需要调用上述函数）
//////////////////////////////////////////////
IplImage* do_FCM(IplImage* src)
{

	double **data;//数据矩阵（一个像素一行）
	double **center;//聚类中心矩阵
	double **u;//样本隶属度矩阵
	int m;//模糊指数
	int row = src->width*src->height;//样本总数
	int col = src->nChannels;//样本属性数（图像通道数）
	cout << "图像尺寸：" << src->width << '*' << src->height << endl;//图像尺寸直接关系到处理数据的规模
	int k;//设定划分的类别
	cout << "请输入模糊指数m：" << endl;
	//cin>>m;
	m = 2;
	cout << "请输入聚类数目k：" << endl;
	//cin>>k;
	k = 2;
	int mum;//算法运行次数
	cout << "设定迭代次数上限" << endl;
	//cin>>mum;
	mum = 100;
	//各次运行结束后的目标函数值
	double *Index = new double[mum];

	//FCM聚类算法开始运行，次数上限mum
	int i, j, t;
	data = new double *[row];
	for (i = 0; i<row; i++)
	{
		data[i] = new double[col];
	}
	t = 0;
	//下面这部分的数据定义主要是为了避免内存耗尽的发生从Update函数中搬下来的
	double **U = new double *[k]; //为了计算方便定义的二维数组U,U[i][j]=pow(u[i][j],m);
	for (j = 0; j<k; j++)
	{
		U[j] = new double[row];
	}
	double *a = new double[col];
	double *b = new double[col];
	double **dis = new double *[k];//聚类中心与样本之间的距离矩阵

	for (i = 0; i<k; i++)
	{
		dis[i] = new double[row];
	}
	////////////////////////////
	//图像数据提取
	for (i = 0; i<src->width; i++)
		for (j = 0; j<src->height; j++)
		{

			for (int t1 = 0; t1<col; t1++)
			{
				data[t][t1] = (double)cvGet2D(src, j, i).val[t1];
			}
			t++;
		}//将图像中的像素各通道强度值存入数组data中，每个像素一行


	double eps = 1e-4;
	int e = 0;//迭代次数循环控制变量

			  //记录连续无改进次数
	int nx = 0;
	//数据极差标准化处理
	data = Standardize(data, row, col);
	/////////////////////以上内容没问题////////////////////////


	//聚类中心及隶属度矩阵的内存分配
	center = new double *[k];
	u = new double *[k];
	for (j = 0; j<k; j++)
	{
		center[j] = new double[col];
		u[j] = new double[row];
	}
	//生成隶属度矩阵（初始化后归一化）
	Initialize(u, k, row);


	//目标函数连续10代无改进，停止该次聚类迭代过程
	for (i = 0; i<mum; i++)
	{

		//聚类迭代过程
		Index[nx] = Update(u, data, center, row, col, k, m, U, dis, a, b);

		if (nx>0 && abs(Index[nx] - Index[nx - 1])<eps)
			e++;
		else
			e = 0;
		nx++;
		cout << nx << '\t' << e << endl;
		if (e >= 10)
			break;

	}


	//绘制聚类后图像
	int t1 = 0;
	double step = 255 / (k - 1);
	IplImage* dst = cvCreateImage(cvGetSize(src), 8, 1);
	for (i = 0; i<src->width; i++)
	{
		for (j = 0; j<src->height; j++)
		{

			CvScalar s;
			s.val[0] = 0;
			for (t = 0; t<k; t++)
			{
				s.val[0] += (255 - t*step)*u[t][t1];
			}
			cvSet2D(dst, j, i, s);
			t1++;
		}

	}
	return(dst);

	//内存释放
	for (j = 0; j<k; j++)
	{
		delete[]center[j];
		delete[]u[j];
	}
	for (j = 0; j<col; j++)
	{
		delete[]data[j];

	}
	delete[]data;
	for (j = 0; j<k; j++)
	{
		delete[]U[j];
	}
	delete[]U;
	delete[]a;
	delete[]b;
	for (i = 0; i<k; i++)
		delete[]dis[i];
	delete[]dis;
	delete[]center;
	delete[]u;
	delete[]Index;

}






///////////////////////////////
///////////////////////////////////////
///==========主=====函=====数=========///
////////////////////////////////////////
int main(int argc, char *argv[])
{
	IplImage* img = 0;
	//const char* filename = "test.tif";
	//const char* filename="Image35[1].jpg";
	//const char* filename="Image35.jpg";//光镜图像
	// const char* filename="finger.jpg";// 对指纹图像进行处理时报错,原因是m_RegionGrowFlag空间设置得小了，改成2*src->width*src->height就能出结果，可是理论上不用乘2就可以啊
	const char* filename = "001.png";

	// load an image  
	//img = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);//载入彩图
	img = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);//载入灰度图或者写成 img=cvLoadImage(filename,0);

	cvNamedWindow("input", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("output", CV_WINDOW_AUTOSIZE);
	cvShowImage("input", img);


	//cvSmooth(img,img,CV_MEDIAN);
	cvSmooth(img, img, CV_GAUSSIAN);//对图像进行高斯滤波后效果更好
	IplImage* out = RegionGrow(0, 0, img, 50);//调用时，初始种子点设为别的点也行，比如设为图像中心点
											  //IplImage* out=RegionGrow(img->height/2,img->width/2,img, 1);
											  //IplImage* out=K_Means(img);
	out = do_FCM(img);
	cvShowImage("output", out);
	// wait for a key
	cvWaitKey(0);
	// release the image
	cvReleaseImage(&out);
	// cvReleaseImage(&hist_image);
	//cvDestroyWindow("hist_imagewindow");
	cvDestroyWindow("input");
	cvDestroyWindow("output");
	return 0;
}


