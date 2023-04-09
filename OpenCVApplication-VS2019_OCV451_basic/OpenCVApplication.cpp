// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iomanip>

using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		cv::waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (cv::waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		cv::waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		cv::waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		cv::waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		cv::waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		cv::waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		cv::waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

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

	//printKernelArray(gauss.first, w);
	//printKernelArray(gauss.second, w);
	//cout << '\n';

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

	//printKernelArray(gauss.first, w);
	//printKernelArray(gauss.second, w);

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
		printf(" 1 - Open image\n");
		printf(" 2 - Gauss Filter Test\n");
		printf(" 3 - Gauss Bidimensional Filter Test\n");

		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");

		printf(" 0 - Exit\n\n");

		printf("Option: ");
		scanf("%d", &op);
		//op = 1;
		switch (op)
		{
			case 1:
				CBIR();
				break;
			case 2:
				gaussFilterTest();
				break;
			case 3:
				gaussBidimensionalFilterTest();
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
		}
	}
	while (op!=0);
	return 0;
}