#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame );
String person_cascade_name;
CascadeClassifier person_cascade;
String window_name = "Capture - Person detection";
int main( int argc, const char** argv )
{
    if (argc != 2) {
      cout << "Add a parameter for the camera number!" << endl;
      return -1;
    }

    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{person_cascade|/home/adam/Documents/opencv-3.2.0/data/haarcascades/haarcascade_upperbody.xml|}");
    cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (person) in a video stream.\n"
            "You can use Haar or LBP features.\n\n";
    parser.printMessage();
    person_cascade_name = parser.get<string>("person_cascade");

    stringstream conv;
    int camSource;
    conv<<argv[1];
    conv>>camSource;

    VideoCapture capture;
    Mat frame, frame_cap, output;
    //-- 1. Load the cascades
    if( !person_cascade.load( person_cascade_name ) ){ printf("--(!)Error loading person cascade\n"); return -1; };
    //-- 2. Read the video stream
    capture.open( camSource );
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 2560);
  	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    int x = 0, y = 0, width = 1280, height = 720;

    while ( capture.read(frame_cap) )
    {
        if( frame_cap.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        //frame_cap(Rect(x,y,width,height)).copyTo(frame);
        frame = frame_cap( Rect(x,y,width,height) );
        cv::resize(frame, output, cv::Size(640,360));
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( output);
        char c = (char)waitKey(10);
        if( c == 27 ) { break; } // escape
    }
    return 0;
}
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> persons, personsFiltered;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect persons
    person_cascade.detectMultiScale( frame_gray, persons, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(30, 30), Size(300,300) );
    int i, j;
    for (i=0; i<persons.size(); i++) {
    	Rect r = persons[i];
    	for (j=0; j<persons.size(); j++)
    		if (j!=i && (r & persons[j])==r)
    			break;
    	  if (j==persons.size())
    			personsFiltered.push_back(r);
    }
    for (i=0; i<personsFiltered.size(); i++) {
    	Rect r = personsFiltered[i];
    	r.x += cvRound(r.width*0.1);
    	r.width = cvRound(r.width*0.8);
    	r.y += cvRound(r.height*0.06);
    	r.height = cvRound(r.height*0.9);
    	rectangle(frame, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
    }
    //-- Show what you got
    imshow( window_name, frame );
}
