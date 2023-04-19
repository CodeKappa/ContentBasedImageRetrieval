// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iomanip>

using namespace std;

int isTest = 0;

void printKernelMatrix(vector<vector<double>> mx, int w)
{
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < w; y++)
		{
			cout << mx[x][y] << ' ';
		}
		cout << '\n';
	}
	cout << '\n';
}

void printKernelArray(vector<double> arr, int w)
{
	for (int i = 0; i < w; i++)
	{
		cout << arr[i] << ' ';
	}
	cout << '\n';
}

bool isInside(Mat img, int row, int col)
{
	int height = img.rows;
	int width = img.cols;

	if (row < 0 || col < 0) return false;
	if (col >= width || row >= height) return false;

	return true;
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void showHistogram(const string& name, vector<double> fdp, const int  hist_cols, const int hist_height, const int x, const int y)
{
	namedWindow(name);
	moveWindow(name, y, x);

	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	double max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (fdp[i] > max_hist)
			max_hist = fdp[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(fdp[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

struct image
{
	Mat mat;
	string name;
	vector<double> hues;
	vector<double> hist;

	image(const Mat mat, const string name, const vector<double> hues, const vector<double> hist)
		: mat(mat), name(name), hues(hues), hist(hist) {}
};

double min3(double a, double b, double c)
{
	return min( min(a, b), c);
}

Mat BGRtoHSI(Mat img)
{
	int height = img.rows;
	int width = img.cols;

	Mat hsiImg = Mat(height, width, CV_64FC3);
	uchar* imgData = img.data;
	double* hsiData = (double *)hsiImg.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int hi = (i * width + j) * 3;
			int gi = i * width + j;

			int R, G, B;
			double r, g, b;
			double H, S, I;

			R = imgData[hi + 2];
			G = imgData[hi + 1];
			B = imgData[hi];

			r = (double)R / 255;
			g = (double)G / 255;
			b = (double)B / 255;

			I = (r + g + b) / 3;

			S = (I == 0 || (R == G && G == B)) ? 0 : (1 - min3(r, g, b) / I);

			H = (S == 0) ? 0 : acos( ((r-g) + (r-b)) / (2 * sqrt((r-g)*(r-g) + (r-b)*(g-b))) );
			H *= 180 / PI;
			H = (b > g) ? 360 - H : H;

			hsiData[hi + 2] = I;
			hsiData[hi + 1] = S;
			hsiData[hi] = H;
		}
	}

	return hsiImg;
}

double getDominantHueValue(Mat img)
{
	int height = img.rows;
	int width = img.cols;

	Mat hsiImg = BGRtoHSI(img);
	double* hsiData = (double *)hsiImg.data;

	double avgHue = -1;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int hi = i * width * 3 + j * 3;

			if (hsiData[hi + 1] != 0) // S == 0
			{
				avgHue += hsiData[hi]; // += H
			}

		}
	}

	if (avgHue != -1)
	{
		avgHue += 1;
		avgHue /= width * height;
	}

	return avgHue;
}

vector<double> getDominantHuesVector(Mat img, int gridSize)
{
	vector<double> dominantHues;

	vector<vector<Mat>> result;
	int width = img.cols / gridSize;
	int height = img.rows / gridSize;
	int startWidth = img.cols - (gridSize - 1) * width;
	int startHeight = img.rows - (gridSize - 1) * height;

	int offset_x = 0;
	int offset_y = 0;

	for (int i = 0; i < gridSize; i++)
	{
		vector<Mat> tmp;
		offset_x = 0;
		for (int j = 0; j < gridSize; j++)
		{
			Rect roi;
			roi.x = offset_x;
			roi.y = offset_y;
			roi.width = (j > 0) ? width : startWidth;
			roi.height = (i > 0) ? height : startHeight;
			offset_x += roi.width;

			Mat crop = img(roi);
			double dhv = getDominantHueValue(crop);

			dominantHues.push_back(dhv);

			//cout << crop.rows << " ";
			//cout << crop.cols << '\n';
		}
		offset_y += (i > 0) ? height : startHeight;
	}

	return dominantHues;
}

vector<pair<double, image>> getMostSimilar(int noValues, int gridSize, vector<double> dominantHues, vector<image> imagedb)
{
	vector<pair<double, image>> similarity;
	for (image img : imagedb)
	{
		int N = 0;
		double SUM = 0, C, D;
		for (int i = 0; i < gridSize * gridSize; i++)
		{
			if (dominantHues[i] != -1 && img.hues[i] != -1)
			{
				N++;
				double aux = abs(dominantHues[i] - img.hues[i]);
				SUM += pow(min(aux, 2 * 180 - aux), 2);
			}
		}

		if (N > 0)
		{
			C = sqrt(N) * 180;
			D = 1.0 * sqrt(SUM) / C;
			similarity.push_back({ 1 - D, img });
		}
	}

	// bubble noValues times to push the most similar images to the end
	for (int i = 0; i < noValues; i++)
	{
		for (int j = 0; j < similarity.size() - i - 1; j++)
		{
			if (similarity[j].first > similarity[j + 1].first)
			{
				iter_swap(similarity.begin() + j, similarity.begin() + j + 1);
			}
		}
	}

	vector<pair<double, image>> result;
	int i = similarity.size() - noValues;
	i = (i > 0) ? i : 0;
	while (i < similarity.size())
	{
		result.push_back(similarity[i++]);
	}
	return result;
}

vector<vector<double>> gaussKernel(double sigma, int w)
{
	double sigma22 = 2 * sigma * sigma;
	//int w = ceil(sigma * 6);
	int x0, y0;
	x0 = y0 = w / 2;
	vector<vector<double>> gauss = vector<vector<double>>(w, vector<double>(w));

	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < w; y++)
		{
			double xx = (x - x0) * (x - x0);
			double yy = (y - y0) * (y - y0);
			gauss[x][y] = (1.0 / PI / sigma22) * exp(-(xx + yy) / sigma22);
		}
	}

	double c = 0;

	for (int i = 0; i < w; i++)
		for (int j = 0; j < w; j++)
			c += gauss[i][j];
	printKernelMatrix(gauss, w);

	for (int i = 0; i < w; i++)
		for (int j = 0; j < w; j++)
			gauss[i][j] /= c;
	printKernelMatrix(gauss, w);

	return gauss;
}

pair<vector<double>, vector<double>> gaussBidimensionalKernel(double sigma, int w)
{
	double sigma22 = 2 * sigma * sigma;
	int x0, y0;
	x0 = y0 = w / 2;
	pair<vector<double>, vector<double>> gauss;
	gauss.first = vector<double>(w);
	gauss.second = vector<double>(w);

	for (int i = 0; i < w; i++)
	{
		double xx = (i - x0) * (i - x0);
		double yy = (i - y0) * (i - y0);

		gauss.first[i] = (1.0 / sqrt(PI * 2) / sigma) * exp(-xx / sigma22);
		gauss.second[i] = (1.0 / sqrt(PI * 2) / sigma) * exp(-yy / sigma22);
	}

	if (isTest == 1)
	{
		printKernelArray(gauss.first, w);
		printKernelArray(gauss.second, w);
		cout << '\n';
	}

	double c = 0;
	for (int i = 0; i < w; i++)
		c += gauss.first[i];
	for (int i = 0; i < w; i++)
		gauss.first[i] / c;

	c = 0;
	for (int i = 0; i < w; i++)
		c += gauss.second[i];
	for (int i = 0; i < w; i++)
		gauss.second[i] / c;

	if (isTest == 1)
	{
		printKernelArray(gauss.first, w);
		printKernelArray(gauss.second, w);
		cout << '\n';
	}

	return gauss;
}

void gaussFilterTest()
{
	int w = 5;
	double sigma = 0.8; // (double) w / 6;

	vector<vector<double>> gauss = gaussKernel(sigma, w);

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		Mat dst = src.clone();

		int height = src.rows;
		int width = src.cols;

		int k = gauss.size() / 2;

		uchar* srcData = src.data;
		uchar* dstData = dst.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int sum = 0, sum1 = 0, sum2 = 0;
				for (int u = 0; u < w; u++)
				{
					for (int v = 0; v < w; v++)
					{
						if (isInside(src, i + u - k, j + v - k))
						{
							int hi = ((i + u - k) * width + (j + v - k)) * 3;
							sum += srcData[hi] * gauss[u][v];
							sum1 += srcData[hi + 1] * gauss[u][v];
							sum2 += srcData[hi + 2] * gauss[u][v];
						}
					}
				}
				int gi = (i * width + j) * 3;
				dstData[gi] = sum;
				dstData[gi + 1] = sum1;
				dstData[gi + 2] = sum2;
			}
		}

		imshow("src", src);
		imshow("dst", dst);

		cv::waitKey();
	}
}

void gaussBidimensionalFilterTest()
{
	int w = 5;
	double sigma = 0.8; // (double) w / 6;

	pair<vector<double>, vector<double>> gauss = gaussBidimensionalKernel(sigma, w);

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		Mat dst = src.clone();
		Mat auxDst = src.clone();

		int height = src.rows;
		int width = src.cols;

		int k = gauss.first.size() / 2;

		uchar* srcData = src.data;
		uchar* dstData = dst.data;
		uchar* auxDstData = dst.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int sum = 0, sum1 = 0, sum2 = 0;
				for (int u = 0; u < w; u++)
				{
					if (isInside(src, i + u - k, j + u - k))
					{
						int hi = ((i + u - k) * width + (j + u - k)) * 3;
						sum +=  (gauss.second[u] * srcData[hi]);
						sum1 += (gauss.second[u] * srcData[hi + 1]);
						sum2 +=  (gauss.second[u] * srcData[hi + 2]);
					}
				}
				int gi = (i * width + j) * 3;
				auxDstData[gi] = sum;
				auxDstData[gi + 1] = sum1;
				auxDstData[gi + 2] = sum2;
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int sum = 0, sum1 = 0, sum2 = 0;
				for (int u = 0; u < w; u++)
				{
					if (isInside(src, i + u - k, j + u - k))
					{
						int hi = ((i + u - k) * width + (j + u - k)) * 3;
						sum += gauss.first[u] * (auxDstData[hi]);
						sum1 += gauss.first[u] * (auxDstData[hi + 1]);
						sum2 += gauss.first[u] * (auxDstData[hi + 2]);
					}
				}
				int gi = (i * width + j) * 3;
				dstData[gi] = sum;
				dstData[gi + 1] = sum1;
				dstData[gi + 2] = sum2;
			}
		}

		imshow("src", src);
		imshow("dst", dst);

		cv::waitKey();
	}
}

Mat gaussBidimensionalFilter(Mat src)
{
	int w = 5;
	double sigma = 0.8; // (double) w / 6;

	pair<vector<double>, vector<double>> gauss = gaussBidimensionalKernel(sigma, w);

	Mat dst = src.clone();
	Mat auxDst;

	int height = src.rows;
	int width = src.cols;

	int k = gauss.first.size() / 2;

	uchar* srcData = src.data;
	uchar* dstData = dst.data;
	uchar* auxDstData = dst.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int sum = 0, sum1 = 0, sum2 = 0;
			for (int u = 0; u < w; u++)
			{
				if (isInside(src, i + u - k, j + u - k))
				{
					int hi = ((i + u - k) * width + (j + u - k)) * 3;
					sum += (gauss.second[u] * srcData[hi]);
					sum1 += (gauss.second[u] * srcData[hi + 1]);
					sum2 += (gauss.second[u] * srcData[hi + 2]);
				}
			}
			int gi = (i * width + j) * 3;
			auxDstData[gi] = sum;
			auxDstData[gi + 1] = sum1;
			auxDstData[gi + 2] = sum2;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int sum = 0, sum1 = 0, sum2 = 0;
			for (int u = 0; u < w; u++)
			{
				if (isInside(src, i + u - k, j + u - k))
				{
					int hi = ((i + u - k) * width + (j + u - k)) * 3;
					sum += gauss.first[u] * (auxDstData[hi]);
					sum1 += gauss.first[u] * (auxDstData[hi + 1]);
					sum2 += gauss.first[u] * (auxDstData[hi + 2]);
				}
			}
			int gi = (i * width + j) * 3;
			dstData[gi] = sum;
			dstData[gi + 1] = sum1;
			dstData[gi + 2] = sum2;
		}
	}

	return dst;
}

double pearsonCorrelationCoefficient(vector<double> x, vector<double> y)
{
	double coefficient = 0;

	double sumX = 0, sumY = 0;
	double sumX2 = 0, sumY2 = 0;
	double sumXY = 0;
	int n = 256;

	for (int i = 0; i < n; i++)
	{
		sumX += x[i];
		sumY += y[i];
		sumXY += x[i] * y[i];
		sumX2 += x[i] * x[i];
		sumY2 += y[i] * y[i];
	}

	coefficient = (double)(n * sumXY - sumX * sumY) 
				/ (sqrt(n * sumX2 - sumX * sumX) * sqrt(n * sumY2 - sumY * sumY));

	return coefficient;
}

vector<double> histogram(Mat img)
{
	vector<int> hist = vector<int>(256, 0);
	int height = img.rows;
	int width = img.cols;

	Mat hsi = BGRtoHSI(img);
	double* hsiData = (double*)hsi.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int hi = (i * width + j) * 3;
			int hue = (double)hsiData[hi] / 360 * 255;
			hist[hue]++;
		}
	}

	vector<double> normHist = vector<double>(256, 0);
	for (int i = 0; i < hist.size(); i++)
	{
		normHist[i] = (double)hist[i] / (width * height);
	}

	return normHist;
}

void displayImage(Mat img, String name, int x, int y)
{
	namedWindow(name);
	moveWindow(name, y, x);
	imshow(name, img);
}

void CBIR()
{
	int noValues = 3;
	int gridSize;
	char folderName[MAX_PATH] = ".\\Images\\Imagini";
	char fname[MAX_PATH];

	vector<image> imagedb;

	cout << "Grid size = ";
	cin >> gridSize;

	//if (openFolderDlg(folderName) == 0)
	//	return;

	int count = 1;

	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		cout << "Processing image " << count++ << '\n';
		Mat src = imread(fname);
		vector<double> dominantHues = getDominantHuesVector(src, gridSize);
		Mat gauss = gaussBidimensionalFilter(src);
		vector<double> hist = histogram(gauss);
		imagedb.emplace_back(image(src, fg.getFoundFileName(), dominantHues, hist));
	}

	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		vector<double> dominantHues = getDominantHuesVector(src, gridSize);
		cout << "Processing similarity... ";
		vector<pair<double, image>> similarity = getMostSimilar(noValues, gridSize, dominantHues, imagedb);
		cout << "done\n";

		Mat srcGauss = gaussBidimensionalFilter(src);
		vector<double> srcHistogram = histogram(srcGauss);

		vector<double> pearsonCoefficients;
		for (pair<double, image> img : similarity)
		{
			double pearson = pearsonCorrelationCoefficient(srcHistogram, img.second.hist);
			pearsonCoefficients.push_back(pearson);
		}

		for (int i = 0; i < noValues; i++)
		{
			for (int j = 0; j < pearsonCoefficients.size() - i - 1; j++)
			{
				if (pearsonCoefficients[j] < pearsonCoefficients[j + 1])
				{
					iter_swap(pearsonCoefficients.begin() + j, pearsonCoefficients.begin() + j + 1);
					iter_swap(similarity.begin() + j, similarity.begin() + j + 1);
				}
			}
		}

		int y = 100;
		displayImage(src, "Selectie", y, 100);
		showHistogram("Histograma", srcHistogram, 256, 200, 400, y);
		for (int i = 0; i < noValues; i++)
		{
			y += 350;
			string nameImg = "Similaritate " + to_string(i + 1);
			string nameHist = "Histograma " + to_string(i + 1);
			displayImage(similarity[i].second.mat, nameImg, 100, y);
			showHistogram(nameHist, similarity[i].second.hist, 256, 200, 400, y);
		}

		cout << setw(40) << "Similaritate 1";
		cout << setw(20) << "Similaritate 2";
		cout << setw(20) << "Similaritate 3" << '\n';
		cout << setw(20) << "Pearson";
		for (int i = 0; i < noValues; i++)
		{
			cout << setw(20) << pearsonCoefficients[i];
		}
		cout << '\n' << setw(20) << "Dominant hues";
		for (int i = 0; i < noValues; i++)
		{
			cout << setw(20) << similarity[i].first;
		}
		cout << '\n';

		cv::waitKey();
		cv::destroyAllWindows();
	}
}

int main()
{
	int op;
	do
	{
		//system("cls");
		cv::destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - CBIR\n");
		printf(" 2 - Gauss Filter Test\n");
		printf(" 3 - Gauss Bidimensional Filter Test\n");

		printf(" 0 - Exit\n\n");

		printf("Option: ");
		scanf("%d", &op);
		//op = 1;
		switch (op)
		{
			case 1:
				isTest = 0;
				CBIR();
				break;
			case 2:
				isTest = 1;
				gaussFilterTest();
				break;
			case 3:
				isTest = 1;
				gaussBidimensionalFilterTest();
				break;
		}
	}
	while (op != 0);
	return 0;
}