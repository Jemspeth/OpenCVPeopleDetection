#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "HOGCascade.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const char* window_name = "Image";

int main(int argc, char **argv)
{
	CascadeClassifier classifier;
	HOGCascadeClassifier hogClassifier;
	//HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, 0.2, true, 5);

	// must check for HOG first, because OpenCV3 will throw an error otherwise
	if(!hogClassifier.load("hogcascade_pedestrians.xml"))
		 if(!classifier.load("hogcascade_pedestrians.xml"))
		     throw std::runtime_error("Could not read file");

	Mat img = imread(argv[1]);
	if(!img.data)
		return -1;
	//resize(img, img, Size(64, 128));
	
	vector<Rect> objects, objfiltered;
	size_t i, j;

	/*
		Parameters: (Mat img, vector<Rect>& objects, double scaleFactor, int minNeighbors,
						int flags, Size minSize, Size maxSize)
		objects-Vector of rectangles where each rectangle contains the detected object.
		scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
		minNeighbors – Parameter specifying how many neighbors each candidate rectangle should
							have to retain it.
		flags – Parameter with the same meaning for an old cascade as in the function.
		minSize – Minimum possible object size. Objects smaller than that are ignored.
		maxSize – Maximum possible object size. Objects larger than that are ignored.

		scaleFactor and minNeighbors can be adjusted for better results depending on image/video
	*/
	hogClassifier.detectMultiScale(img, objects, 1.05, 2, 0, Size(10, 10), Size(1000, 1000));

	for (i=0; i<objects.size(); i++) {
		Rect r = objects[i];
		for (j=0; j<objects.size(); j++)
			if (j!=i && (r & objects[j])==r)
				break;
		if (j==objects.size())
			objfiltered.push_back(r);
	}
	for (i=0; i<objfiltered.size(); i++) {
		Rect r = objfiltered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.06);
		r.height = cvRound(r.height*0.9);
		rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
	}

	namedWindow(window_name, WINDOW_AUTOSIZE);
	moveWindow(window_name, 100, 100);
	imshow(window_name, img);
	
	waitKey(0);
	return 0;
}
