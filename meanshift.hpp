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
	double xPosition;
	double yPosition;
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

typedef struct cluster
{
	double vec;
	std::vector<pixel> points;
} cluster;

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
						ANNidxArray nnIdx = findNearestNeighbors(currentMatrix[(y * width) + x], dataPts, kdTree);

						// DO THE MATH!
						double xNumerator  = 0.0;
						double denominator = 0.0;
						double yNumerator  = 0.0;
						double lNumerator  = 0.0;
						double uNumerator  = 0.0;
						double vNumerator  = 0.0;

						for (int i = 0; i < k; i++) // iterating over each neighbor
						{
							double moduleXs = sqrt(pow(dataPts[nnIdx[i]][0] - currentMatrix[(y* width) + x].xPosition, 2) 
								+ pow(dataPts[nnIdx[i]][1] - currentMatrix[(y* width) + x].yPosition, 2));
							double moduleXr = sqrt(pow(dataPts[nnIdx[i]][2] - currentMatrix[(y* width) + x].lColor, 2) 
								+ pow(dataPts[nnIdx[i]][3] - currentMatrix[(y* width) + x].uColor, 2) 
								+ pow(dataPts[nnIdx[i]][4] - currentMatrix[(y* width) + x].vColor, 2));
							double xsPow    = - (pow(moduleXs / hs, 2));
							double xrPow    = - (pow(moduleXr / hr, 2));

							double den = exp (xsPow) * exp (xrPow);
						
							xNumerator  += dataPts[nnIdx[i]][0] * den;
							yNumerator  += dataPts[nnIdx[i]][1] * den;
							lNumerator  += dataPts[nnIdx[i]][2] * den;
							uNumerator  += dataPts[nnIdx[i]][3] * den;
							vNumerator  += dataPts[nnIdx[i]][4] * den;
							denominator += den;
						}
						
						double vec = sqrt(pow(xNumerator/denominator - currentMatrix[(y *  width) + x].xPosition, 2) + 
							              pow(yNumerator/denominator - currentMatrix[(y *  width) + x].yPosition, 2) +
							              pow(lNumerator/denominator - currentMatrix[(y *  width) + x].lColor,    2) +
							              pow(uNumerator/denominator - currentMatrix[(y *  width) + x].uColor,    2) +
							              pow(vNumerator/denominator - currentMatrix[(y *  width) + x].vColor,    2));


						currentMatrix[(y *  width) + x].xPosition = xNumerator / denominator;
						currentMatrix[(y *  width) + x].yPosition = yNumerator / denominator;
						currentMatrix[(y *  width) + x].lColor    = lNumerator / denominator;
						currentMatrix[(y *  width) + x].uColor    = uNumerator / denominator;
						currentMatrix[(y *  width) + x].vColor    = vNumerator / denominator;

						dataPts[(y * width) + x][0] = currentMatrix[(y *  width) + x].xPosition;
						dataPts[(y * width) + x][1] = currentMatrix[(y *  width) + x].yPosition;
						dataPts[(y * width) + x][2] = currentMatrix[(y *  width) + x].lColor;
						dataPts[(y * width) + x][3] = currentMatrix[(y *  width) + x].uColor;
						dataPts[(y * width) + x][4] = currentMatrix[(y *  width) + x].vColor;

						// // std::cout << "    " << x << "," << y << std::endl;
						// if (x == 25 && y == 82)
						// {
						// 	std::cout << xNumerator/denominator << ", " << currentMatrix[(y * width) + x].xPosition << std::endl;
						// 	// std::cout << "    "<< vec << std::endl;
						// }

						if (vec <= eps)
						{
							std::cout << kernelsChanging << std::endl;
							kernelsChanging--;
							currentMatrix[(y *  width) + x].isMoving = false;
						}

						delete [] nnIdx;
					}
				}
			}
		}

		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				pixel p = currentMatrix[(y * width) + x];

				img(x,y,0,0) = img(p.xPosition, p.yPosition, 0, 0); // r
				img(x,y,0,1) = img(p.xPosition, p.yPosition, 0, 1); // g
				img(x,y,0,2) = img(p.xPosition, p.yPosition, 0, 2); // b
			}
		}

		img.save("images/output/result.png");

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

	// pixel bilinearInterpolation(double x, double y, int width, int height)
	// {
	// 	pixel a,b,c,d;
	// 	a.xPosition = floor(x);
	// 	a.yPosition = floor(y);

	// 	b.xPosition = floor(x);
	// 	b.yPosition = ceil(y);
	// 	if (b.yPosition == height)
	// 		b.yPosition--;

	// 	c.xPosition = ceil(x);
	// 	if (c.xPosition == width)
	// 		c.xPosition--;
	// 	c.yPosition = floor(y);

	// 	d.xPosition = ceil(x);
	// 	if (d.xPosition == width)
	// 		d.xPosition--;
	// 	d.yPosition = ceil(y);
	// 	if (d.yPosition == height)
	// 		d.yPosition--;

	// 	double distanceA = sqrt(pow((x - a.xPosition), 2) + pow ((y - a.yPosition), 2));
	// 	double distanceB = sqrt(pow((x - b.xPosition), 2) + pow ((y - b.yPosition), 2));
	// 	double distanceC = sqrt(pow((x - c.xPosition), 2) + pow ((y - c.yPosition), 2));
	// 	double distanceD = sqrt(pow((x - d.xPosition), 2) + pow ((y - d.yPosition), 2));

	// 	if (distanceA <= distanceB && distanceA <= distanceC && distanceA <= distanceD)
	// 		return a;
	// 	if (distanceB <= distanceA && distanceB <= distanceC && distanceB <= distanceD)
	// 		return b;
	// 	if (distanceC <= distanceA && distanceC <= distanceB && distanceC <= distanceD)
	// 		return c;
	// 	if (distanceD <= distanceA && distanceD <= distanceB && distanceD <= distanceC)
	// 		return d;
	// }

private:
	imageMatrix image;
	int dimension = 5; // dimension of the space (default = 2)
	int k         = 60; // number of neighbors
	double eps    = 0.008856;
};
