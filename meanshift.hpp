#include "libs/CImg.h"
#include <iostream>
#include <algorithm>
#include <ANN/ANN.h>
#include <cstdlib>						// C standard library
#include <cstdio>						// C I/O (for sscanf)
#include <cstring>						// string manipulation
#include <math.h>

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
	bool isMoving;
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
		image.width  = width;
		image.height = height;

		pixel *currentMatrix = new pixel[width*height];

		// for each pixel
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				image.matrix[x][y].xPosition = (double) x;
				image.matrix[x][y].yPosition = (double) y;
				image.matrix[x][y].red       = img(x,y,0,0);
				image.matrix[x][y].green     = img(x,y,0,1);
				image.matrix[x][y].blue      = img(x,y,0,2);
			}
		}

		changeColorSpace();

		int maxPts = width * height;                 // maximum number of data points (default = 1000)
		ANNpointArray dataPts = annAllocPts(maxPts, dimension);	 // allocate data points
		generateDataPts(dataPts, currentMatrix);

		std::cout << "oi" << std::endl;

		ANNkd_tree*	  kdTree;			            // search structure
		kdTree = new ANNkd_tree(					// build search structure
					dataPts,					    // the data points
					maxPts,						    // number of points
					dimension);						// dimension of space

		int kernelsChanging = maxPts;
		while (kernelsChanging > 0)
		{
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					if (currentMatrix[(y *  width) + x].isMoving)
					{
						ANNidxArray nnIdx = findNearestNeighbors(image.matrix[x][y], dataPts, kdTree);

						// DO THE MATH!
						double xNumerator  = 0.0;
						double denominator = 0.0;
						double yNumerator  = 0.0;
						double lNumerator  = 0.0;
						double uNumerator  = 0.0;
						double vNumerator  = 0.0;
						
						for (int i = 0; i < k; i++) // iterating over each neighbor
						{
							double moduleXs = sqrt(pow(dataPts[nnIdx[i]][0], 2) + pow(dataPts[nnIdx[i]][1], 2));
							double moduleXr = sqrt(pow(dataPts[nnIdx[i]][2], 2) + pow(dataPts[nnIdx[i]][3], 2) + pow(dataPts[nnIdx[i]][4], 2));
							double xsPow = - (pow(moduleXs / hs, 2));
							double xrPow = - (pow(moduleXr / hr, 2));

							double den = exp (xsPow) * exp (xrPow);

							xNumerator += dataPts[nnIdx[i]][0] * den;
							denominator += den;
							yNumerator += dataPts[nnIdx[i]][1] * den;
							lNumerator += dataPts[nnIdx[i]][2] * den;
							uNumerator += dataPts[nnIdx[i]][3] * den;
							vNumerator += dataPts[nnIdx[i]][4] * den;
							// std::cout << dataPts[nnIdx[i]][4] << std::endl;
						}
						currentMatrix[(y *  width) + x].xPosition += xNumerator / denominator;
						currentMatrix[(y *  width) + x].yPosition += yNumerator / denominator;
						currentMatrix[(y *  width) + x].lColor    += lNumerator / denominator;
						currentMatrix[(y *  width) + x].uColor    += uNumerator / denominator;
						currentMatrix[(y *  width) + x].vColor    += vNumerator / denominator;
						
						double vec = sqrt(pow(xNumerator/denominator, 2) + 
							pow(yNumerator/denominator, 2) +
							pow(lNumerator/denominator, 2) +
							pow(uNumerator/denominator, 2) +
							pow(vNumerator/denominator, 2));

						if (vec <= eps)
						{
							kernelsChanging--;
							currentMatrix[(y *  width) + x].isMoving = false;
						}

						delete [] nnIdx;
					}
				}
			}

			break;
		}

		delete [] currentMatrix;
	}

	void generateDataPts (ANNpointArray & dataPts, pixel * currentMatrix)
	{
		// insert all points in dataPts
		for (int y = 0; y < image.height; y++)	
		{
			for (int x = 0; x < image.width; x++)
			{
				ANNpoint point = annAllocPt(dimension);			    // allocate query point
				point[0]       = (double) x;
				point[1]       = (double) y;
				point[2]       = image.matrix[x][y].lColor;
				point[3]       = image.matrix[x][y].uColor;
				point[4]       = image.matrix[x][y].vColor;
				
				dataPts[(y * image.width) + x] = annCopyPt(dimension, point);

				currentMatrix[(y * image.width) + x].xPosition = point[0];
				currentMatrix[(y * image.width) + x].yPosition = point[1];
				currentMatrix[(y * image.width) + x].lColor    = point[2];
				currentMatrix[(y * image.width) + x].uColor    = point[3];
				currentMatrix[(y * image.width) + x].vColor    = point[4];
				currentMatrix[(y * image.width) + x].isMoving  = true;
			}
		}
	}

	ANNidxArray findNearestNeighbors (pixel p, ANNpointArray & dataPts, ANNkd_tree * kdTree)
	{
		ANNpoint	  queryPt;				            // query point
		ANNidxArray	  nnIdx;				            // near neighbor indices
		ANNdistArray  dists;				            // near neighbor distances

		queryPt = annAllocPt(dimension);			    // allocate query point
		nnIdx   = new ANNidx[k];						// allocate near neigh indices
		dists   = new ANNdist[k];						// allocate near neighbor dists

		queryPt[0] = p.xPosition;
		queryPt[1] = p.yPosition;
		queryPt[2] = p.lColor;
		queryPt[3] = p.uColor;
		queryPt[4] = p.vColor;

		kdTree->annkSearch(						// search
				queryPt,						// query point
				k,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				eps);							// error bound

    	delete [] dists;
		annClose();								// done with ANN

		return nnIdx;
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

		for (int x = 0; x < image.width; x++)
		{
			for (int y = 0; y < image.height; y++)
			{
				double rgb[3][1] = { image.matrix[x][y].red, image.matrix[x][y].green, image.matrix[x][y].blue };

				double ** xyzAnswer = multiplyMatrixColor(rgbToxyz, rgb);

				double yParam = (xyzAnswer[1][0] / white[1]) / 255.0;
				double L;
				if (yParam > eps)
				{
					L = (116.0 * pow(yParam, 1.0/3)) - 16.0;
				}
				else
				{
					L = 903.3 * y;
				}

				double u;
				double v;
				double denominator = xyzAnswer[0][0] + 15.0 * xyzAnswer[1][0] + 3.0 * xyzAnswer[2][0];
				if (denominator == 0.0)
				{
					u = 4.0;
					v = 9.0/15;
				}
				else
				{
					u = 4.0 * xyzAnswer[0][0] / denominator;
					v = 9.0 * xyzAnswer[1][0] / denominator;
				}

				u = 13.0 * L * (u - 0.19784977571475);
				v = 13.0 * L * (v - 0.46834507665248);

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
	int k         = 60; // number of neighbors
	double eps    = 0.008856;
};
