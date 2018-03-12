
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
//����������
//////////////////////////
IplImage* RegionGrow(int x, int y, IplImage *src, int gate) //�����x��yָ��x��y��
{
	//8�����Ӧ��������
	int indexx[] = { -1,-1,-1,0,0,1,1,1 };
	int indexy[] = { -1,0,1,-1,1,-1,0,1 };
	int k;//ѭ�����Ʊ���

		  //����ָ��(һά����)�Դ洢����
	int *m_RegionGrowX;
	int *m_RegionGrowY;
	int *m_RegionGrowFlag;
	//���ٿռ�
	m_RegionGrowX = new int[8 * src->width*src->height];
	m_RegionGrowY = new int[8 * src->width*src->height];
	m_RegionGrowFlag = new int[8 * src->width*src->height];
	for (int i = 0; i<src->height; i++)
		for (int j = 0; j<src->width; j++)
		{
			m_RegionGrowFlag[i*src->widthStep + j] = 0;

		}
	//����������յ�
	int m_Start;
	int m_End;
	//����ֵ
	m_Start = 0;
	m_End = 0;
	//����ʼ���ӵ������������
	m_RegionGrowX[m_End] = x;
	m_RegionGrowY[m_End] = y;


	//��ǰ�����꣨���ĵ����꣩Ҳ�������ӵ�
	int m_CurrX;
	int m_CurrY;
	//�µĵ����꣨��������꣩
	int m_NewX;
	int m_NewY;
	while (m_Start<(src->width*src->height) - 1)
	{

		while (m_Start <= m_End)
		{
			//��ǰ�����긳ֵ�����Ե���������������Ѱ�һҶȽӽ��ĵ�
			m_CurrX = m_RegionGrowX[m_Start];
			m_CurrY = m_RegionGrowY[m_Start];
			for (k = 0; k<8; k++)
			{
				m_NewX = m_CurrX + indexx[k];
				m_NewY = m_CurrY + indexy[k];
				if ((m_NewX<src->height) && (m_NewY<src->width) && (m_NewX >= 0) && (m_NewY >= 0))  //�ж��������ͼ��Χ�ڲ��ܽ�����������
				{
					uchar temp = uchar((src->imageData + src->widthStep*m_CurrX)[m_CurrY]);
					uchar temp1 = uchar((src->imageData + src->widthStep*m_NewX)[m_NewY]);
					int qq;
					qq = int(temp - temp1);
					int pp;
					pp = m_NewX*src->width + m_NewY;
					int mm;
					mm = m_RegionGrowFlag[pp];
					// �����������ж�����(m_NewX,m_NewY)�͵�ǰ����(m_CurrX,m_CurrY) ����ֵ��ľ���ֵ<=gate���������
					if ((mm == 0) && (abs(qq) <= gate))
					{
						m_End++;
						//�����������������������Ա����������´���������㣨���ӣ�
						m_RegionGrowX[m_End] = m_NewX;
						m_RegionGrowY[m_End] = m_NewY;
						m_RegionGrowFlag[m_NewX*src->width + m_NewY] = 1;//����Ѷ����ص�
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
			//����������Ҷ�ֵ��Ϊ0��������Ϊ255
			if (m_RegionGrowFlag[i*src->width + j] == 1)
				(dst->imageData + dst->widthStep*i)[j] = 0;
			else
				(dst->imageData + dst->widthStep*i)[j] = 255;
		}
	delete[]m_RegionGrowFlag;
	m_RegionGrowFlag = NULL;
	return(dst);

};


//K��ֵ����
///////////////////////////////
IplImage* K_Means(IplImage* src)
{
	int i, j, k = 0, value;
	int nCuster = 3;//����������Ŀ
					//��������cluster���Ա�־ÿ��������Ӧ�����ȡֵ��Χ0,1,2...nCuster-1;
	CvMat* clusters = cvCreateMat(src->height*src->width, 1, CV_32SC1);//Opencv�ڲ�����cvKMeans2Ҫ��label���������CV_32SC1��
	CvMat* samples = cvCreateMat(src->height*src->width, 1, CV_32FC2);//Ҫ��sampels���������CV_32FC2��
	IplImage* dst = cvCreateImage(cvGetSize(src), 8, 1);
	for (i = 0; i<src->width; i++)
		for (j = 0; j<src->height; j++)
		{
			CvScalar s;
			//��ȡͼ��������ص����ͨ��ֵ(BGR)
			s.val[0] = (float)cvGet2D(src, j, i).val[0];
			s.val[1] = (float)cvGet2D(src, j, i).val[1];
			s.val[2] = (float)cvGet2D(src, j, i).val[2];
			cvSet2D(samples, k++, 0, s);

		}

	cvKMeans2(samples, nCuster, clusters, cvTermCriteria(CV_TERMCRIT_ITER, 100, 1.0));
	//���ƾ�����ͼ��
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


//ģ��C�����㷨׼������
///////////////////////////////
double** Standardize(double **data, int row, int col)
{
	int i, j;
	double *a = new double[col];//����ÿ�����ֵ
	double *b = new double[col];//����ÿ����Сֵ
	double *c = new double[row];//������ʱ�洢����ĳһ��Ԫ��
	for (i = 0; i<col; i++)
	{
		//ȡ�����ݾ���ĸ���Ԫ��
		for (j = 0; j<row; j++)
		{
			c[j] = data[j][i];
		}
		a[i] = c[0];
		b[i] = c[0];
		for (j = 0; j<row; j++)
		{
			//�����ֵ
			if (c[j]>a[i])
				a[i] = c[j];
			//����Сֵ
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
	cout << "������ݼ����׼������>>>>>>>>>\n";
	delete[]a;
	delete[]b;
	delete[]c;

	return(data);

}
//�������������Ⱦ���
void Initialize(double **u, int k, int row)//kΪ������
{
	int i, j;
	//��ʼ�����������Ⱦ���
	srand((unsigned)time(0));
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			u[i][j] = (double)rand() / RAND_MAX;//������ȡֵ��Χ��Ϊ0~1
		}//rand()��������0~RANDN_MAX֮���һ��α�����
	}
	//���������ݹ�һ��
	double *sum = new double[row];//�����Ⱦ���ÿ�еĺ�
	for (j = 0; j<row; j++)//������row��ʾ��������ƴд���������Ϊ�����Ⱦ������������data�����������row��data�������
	{
		double dj = 0;
		for (i = 0; i<k; i++)
		{
			dj += u[i][j];
		}
		sum[j] = dj;//�����Ⱦ������֮��

	}
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			u[i][j] /= sum[j];//��һ��
		}
	}

	cout << "���������Ⱦ�������>>>>>>>>>>>>>" << endl;
	delete[]sum;
}
//��������
double Update(double **u, double **data, double **center, int row, int col, int k, int m, double **U, double **dis, double *a, double *b)
{
	int i, j, t;
	/*double **U=new double *[k];
	for(j=0;j<k;j++)
	{
	U[j]=new double[row];
	}*/
	double si = 0;//center(i,j) �ķ���
	double sj = 0;//center(i,j) �ķ�ĸ
				  //���������Ⱦ������������ģ��μ�����3.4ʽ
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			U[i][j] = pow(u[i][j], m);//mΪģ��ָ����Խ��Խģ����ԽСԽ�ӽ�K��ֵ����

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
	//���������������i�ֱ����е�j�ľ������dis(i,j)
	/*double *a=new double[col];
	double *b=new double[col];
	double **dis=new double *[k];//��������������֮��ľ������

	for(i=0;i<k;i++)
	{
	dis[i]=new double[row];
	}*/
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<col; j++)
			a[j] = center[i][j];//�ݴ�һ��������
		for (j = 0; j<row; j++)
		{
			for (t = 0; t<col; t++)
				b[t] = data[j][t];//�ݴ�һ������
			double d = 0;
			//�����������������֮��ľ���
			for (t = 0; t<col; t++)
			{
				d += (a[t] - b[t])*(a[t] - b[t]);//dΪһ��������һ��������ŷ����þ����ƽ��
			}
			dis[i][j] = sqrt(d);//ŷ����þ���
		}
	}
	//�����Ⱦ���ĸ��£��μ����ĵ�3.5ʽ
	for (i = 0; i<k; i++)
	{
		for (j = 0; j<row; j++)
		{
			double temp = 0;
			for (t = 0; t<k; t++)
			{
				//ģ��ָ��Ϊm
				temp += pow(dis[i][j] / dis[t][j], 2 / (m - 1));
			}
			u[i][j] = 1 / temp;
		}
	}
	//����FCM�ļ�ֵ������Ŀ�꺯�������������Ч�����۲����μ�������3.2ʽ
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
	//�ڴ��ͷ�
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
//ģ��C��ֵ�����㷨����Ҫ��������������
//////////////////////////////////////////////
IplImage* do_FCM(IplImage* src)
{

	double **data;//���ݾ���һ������һ�У�
	double **center;//�������ľ���
	double **u;//���������Ⱦ���
	int m;//ģ��ָ��
	int row = src->width*src->height;//��������
	int col = src->nChannels;//������������ͼ��ͨ������
	cout << "ͼ��ߴ磺" << src->width << '*' << src->height << endl;//ͼ��ߴ�ֱ�ӹ�ϵ���������ݵĹ�ģ
	int k;//�趨���ֵ����
	cout << "������ģ��ָ��m��" << endl;
	//cin>>m;
	m = 2;
	cout << "�����������Ŀk��" << endl;
	//cin>>k;
	k = 2;
	int mum;//�㷨���д���
	cout << "�趨������������" << endl;
	//cin>>mum;
	mum = 100;
	//�������н������Ŀ�꺯��ֵ
	double *Index = new double[mum];

	//FCM�����㷨��ʼ���У���������mum
	int i, j, t;
	data = new double *[row];
	for (i = 0; i<row; i++)
	{
		data[i] = new double[col];
	}
	t = 0;
	//�����ⲿ�ֵ����ݶ�����Ҫ��Ϊ�˱����ڴ�ľ��ķ�����Update�����а�������
	double **U = new double *[k]; //Ϊ�˼��㷽�㶨��Ķ�ά����U,U[i][j]=pow(u[i][j],m);
	for (j = 0; j<k; j++)
	{
		U[j] = new double[row];
	}
	double *a = new double[col];
	double *b = new double[col];
	double **dis = new double *[k];//��������������֮��ľ������

	for (i = 0; i<k; i++)
	{
		dis[i] = new double[row];
	}
	////////////////////////////
	//ͼ��������ȡ
	for (i = 0; i<src->width; i++)
		for (j = 0; j<src->height; j++)
		{

			for (int t1 = 0; t1<col; t1++)
			{
				data[t][t1] = (double)cvGet2D(src, j, i).val[t1];
			}
			t++;
		}//��ͼ���е����ظ�ͨ��ǿ��ֵ��������data�У�ÿ������һ��


	double eps = 1e-4;
	int e = 0;//��������ѭ�����Ʊ���

			  //��¼�����޸Ľ�����
	int nx = 0;
	//���ݼ����׼������
	data = Standardize(data, row, col);
	/////////////////////��������û����////////////////////////


	//�������ļ������Ⱦ�����ڴ����
	center = new double *[k];
	u = new double *[k];
	for (j = 0; j<k; j++)
	{
		center[j] = new double[col];
		u[j] = new double[row];
	}
	//���������Ⱦ��󣨳�ʼ�����һ����
	Initialize(u, k, row);


	//Ŀ�꺯������10���޸Ľ���ֹͣ�ôξ����������
	for (i = 0; i<mum; i++)
	{

		//�����������
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


	//���ƾ����ͼ��
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

	//�ڴ��ͷ�
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
///==========��=====��=====��=========///
////////////////////////////////////////
int main(int argc, char *argv[])
{
	IplImage* img = 0;
	//const char* filename = "test.tif";
	//const char* filename="Image35[1].jpg";
	//const char* filename="Image35.jpg";//�⾵ͼ��
	// const char* filename="finger.jpg";// ��ָ��ͼ����д���ʱ����,ԭ����m_RegionGrowFlag�ռ����õ�С�ˣ��ĳ�2*src->width*src->height���ܳ���������������ϲ��ó�2�Ϳ��԰�
	const char* filename = "001.png";

	// load an image  
	//img = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);//�����ͼ
	img = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);//����Ҷ�ͼ����д�� img=cvLoadImage(filename,0);

	cvNamedWindow("input", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("output", CV_WINDOW_AUTOSIZE);
	cvShowImage("input", img);


	//cvSmooth(img,img,CV_MEDIAN);
	cvSmooth(img, img, CV_GAUSSIAN);//��ͼ����и�˹�˲���Ч������
	IplImage* out = RegionGrow(0, 0, img, 50);//����ʱ����ʼ���ӵ���Ϊ��ĵ�Ҳ�У�������Ϊͼ�����ĵ�
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


