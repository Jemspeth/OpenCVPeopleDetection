//./HogSVM ~/Downloads/INRIAPerson/train_64x128_H96/pos/ ~/Downloads/INRIAPerson/train_64x128_H96/pos.lst 

#include <iostream>
#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace ml;

const char *window = "Image";

void load_images(string directory, string list, vector<Mat> &img_lst);
void compute_hog(const vector<Mat> &img_lst, vector<Mat> &gradient_lst, const Size &size);
void convert_to_LibSVM(const vector<Mat> &gradient_lst, Mat &train_data);
void train_svm(const vector<Mat> &gradient_lst, const vector<int> &labels);

int main(int argc, char **argv)
{
	string pos_dir = argv[1];
	string pos_lst = argv[2];
	vector<Mat> pos_imgs;
	load_images(pos_dir, pos_lst, pos_imgs);

	string neg_dir = argv[3];
	string neg_lst = argv[4];
	vector<Mat> neg_imgs;
	load_images(neg_dir, neg_lst, neg_imgs);


	Mat trainData;
	Mat labels;

	namedWindow(window, WINDOW_AUTOSIZE);
	imshow(window, pos_lst[300]);
	waitKey(0);
	return 0;
}

void load_images(string directory, string list, vector<Mat> &img_lst) {
	ifstream infile;
	infile.open(list.c_str());

	if(!infile.is_open()) {
		cerr << "Unable to open file." << endl;
		exit(-1);
	}

	bool end_of_file = false;
	string line;

	while(!end_of_file) {
		getline(infile, line);
		if(line.empty()) {
			end_of_file = true;
			break;
		}
		Mat img = imread((directory+line).c_str());
		if(img.empty())
			continue;

		img_lst.push_back(img.clone());
	}
}

void compute_hog(const vector<Mat> &img_lst, vector<Mat> &gradient_lst, const Size &size) {
	HOGDescriptor hog(Size(64, 128), 
							Size(8, 8), 
							Size(4, 4), 
							Size(4, 4), 
							9, 
							1, 
							-1, 
							0,		//L2-Hys
							0.2, 
							1, 
							64, 
							1);
	hog.winSize = size;
	Mat gray;
	vector<Point> location;
	vector<float> descriptors;

	vector<Mat>::const_iterator img = img_lst.begin();
	vector<Mat>::const_iterator end = img_lst.end();

	for(; img != end; ++img) {
		cvtColor(*img, gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, hog.blockStride, Size(0, 0), location);
		gradient_lst.push_back( Mat(descriptors).clone() );
	}	
}

void convert_to_LibSVM(const vector<Mat> &gradient_lst, Mat &train_data) {

}

void train_svm(const vector<Mat> &gradient_lst, const vector<int> &labels) {
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::ONE_CLASS);
	svm->setKernel(SVM::LINEAR);
	svm->setC(25);
	svm->setGamma(2);
}
