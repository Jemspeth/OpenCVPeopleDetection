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

int main(int argc, char **argv)
{
	HOGDescriptor hog(Size(64, 128), 
										Size(8, 8), 
										Size(4, 4), 
										Size(4, 4), 
										9, 
										1, 
										-1, 
										0, 
										0.2, 
										1, 
										64, 
										1);

	vector<float> descriptors;
	//hog.compute(img, descriptors);

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::ONE_CLASS);
	svm->setKernel(SVM::LINEAR);
	svm->setC(25);
	svm->setGamma(2);


	Mat trainData;
	Mat labels;

	vector<Mat> pos_lst;
	string pos_imgs = argv[1];
	string pos_file = argv[2];
	load_images(pos_imgs, pos_file, pos_lst);

	namedWindow(window, WINDOW_AUTOSIZE);
	imshow(window, pos_lst[300]);
	waitKey(0);
	return;
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
