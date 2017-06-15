#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "HOGCascade.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
cv::CascadeClassifier classifier;
cv::HOGCascadeClassifier HOGClassifier;

// must check for HOG first, because OpenCV3 will throw an error otherwise
if(!HOGClassifier.load("hogcascade_pedestrians.xml"))
  if(!classifier.load("hogcascade_pedestrians.xml"))
    throw std::runtime_error("Could not read file");



  return 0;
}
