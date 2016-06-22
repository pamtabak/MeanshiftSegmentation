#include "libs/CImg.h"
#include <iostream>
#include <algorithm>
#include <ANN/ANN.h>
#include <cstdlib>						// C standard library
#include <cstdio>						// C I/O (for sscanf)
#include <cstring>						// string manipulation

#define cimg_use_magick

using namespace cimg_library;

typedef struct pixel
{
	int xPosition;
	int yPosition;
	double red;
	double green;
	double blue;
	// double xColor;
	// double yColor;
	// double zColor;
	double lColor;
	double uColor;
	double vColor;
} pixel;

typedef struct imageMatrix
{
	int width;
	int height;
	pixel **matrix;
} imageMatrix;

class Meanshift
{
public:
	Meanshift()
	{

	}
	~Meanshift()
	{
		delete image.matrix;
	}

	void initializeImage(int width, int height)
	{
		image.matrix = new pixel*[width];
		for (int i = 0; i < width; i++)
		{
			image.matrix[i] = new pixel[height];
		}
	}

	void filtering (CImg<double> img, double hs, double hr)
	{
		int width    = img.width();
		int height   = img.height();
		int spectrum = 3; 
		int depth    = 1;

		initializeImage(width, height);

		findNearestNeighboors(width, height, 50);

		// // for each pixel
		// for (int x = 0; x < width; x++)
		// {
		// 	for (int y = 0; y < height; y++)
		// 	{
		// 		image.matrix[x][y].xPosition = x;
		// 		image.matrix[x][y].yPosition = y;
		// 		image.matrix[x][y].red       = img(x,y,0,0);
		// 		image.matrix[x][y].green     = img(x,y,0,1);
		// 		image.matrix[x][y].blue      = img(x,y,0,2);
		// 	}
		// }

		changeColorSpace();
	}

	void findNearestNeighboors (int width, int height, int k)
	{
		std::cout << "findNearestNeighboors" << std::endl;
		int			  nPts;					// actual number of data points
		ANNpointArray dataPts;				// data points
		ANNpoint	  queryPt;				// query point
		ANNidxArray	  nnIdx;				// near neighbor indices
		ANNdistArray  dists;				// near neighbor distances
		ANNkd_tree*	  kdTree;			    // search structure

		int maxPts = width * height;// maximum number of data points (default = 1000)

		// queryPt = annAllocPt(dimension);					    // allocate query point
		// dataPts = annAllocPts(maxPts, dimension);		// allocate data points
		// nnIdx   = new ANNidx[k];						// allocate near neigh indices
		// dists   = new ANNdist[k];						// allocate near neighbor dists

		nPts = 0;									// read data points
		// kdTree = new ANNkd_tree(					// build search structure
		// 			dataPts,					    // the data points
		// 			maxPts,						    // number of points
		// 			dimension);						// dimension of space
	}

	// color range [0,1]
	void changeColorSpace ()
	{
		// RGB -> XYZ
		// XYZ -> Luv

		imageMatrix xyzMatrix;
		xyzMatrix.matrix = new pixel*[image.width];
		for (int i = 0; i < image.width; i++)
		{
			xyzMatrix.matrix[i] = new pixel[image.height];
		}

		double rgbToxyz[3][3] = {{0.412453, 0.35758, 0.180423}, {0.212671, 0.71516, 0.072169}, {0.019334, 0.119193, 0.950227}};
		double white[3]       = {0.95047, 1.00, 1.08883};
		double eps            = 0.008856;

		for (int x = 0; x < image.width; x++)
		{
			for (int y = 0; y < image.height; y++)
			{
				double rgb[3][1] = { image.matrix[x][y].red / 255.0, image.matrix[x][y].green / 255.0, image.matrix[x][y].blue / 255.0 };

				double ** xyzAnswer = multiplyMatrixColor(rgbToxyz, rgb);

				double yParam = (xyzAnswer[2][0] / white[1]) / 255.0;
				double L;
				if (yParam > eps)
				{
					L = (116 * pow(yParam, 1/3)) - 16;
				}
				else
				{
					L = 903.3 * y;
				}

				double u;
				double v;
				double denominator = xyzAnswer[0][0] + 15 * xyzAnswer[1][0] + 3*xyzAnswer[2][0];
				if (denominator == 0.0)
				{
					u = 4.0;
					v = 9.0/15;
				}
				else
				{
					u = 4 * xyzAnswer[0][0];
					v = 9 * xyzAnswer[1][0];
				}

				u = 13 * L * (u - 0.19784977571475);
				v = 13 * L * (v - 0.46834507665248);

				image.matrix[x][y].lColor = L;
				image.matrix[x][y].uColor = u;
				image.matrix[x][y].vColor = v;

				delete xyzAnswer;
			}
		}

		delete xyzMatrix.matrix;
	}

	double ** multiplyMatrixColor (double a[3][3], double b[3][1])
	{
		double ** answer;
		answer = new double*[3];
		for (int i = 0; i < 3; i++)
		{
			answer[i] = new double[1];
		}

		for (int x = 0; x < 3; x++)
		{
			for (int y = 0; y < 1; y++)
			{
				for (int z = 0; z < 3; z++)
				{
					answer[x][y] += a[x][z]*b[z][y];
				}
			}
		}

		return answer;
	}

private:
	imageMatrix image;
	int dimension = 5; // dimension of the space (default = 2)
};
