#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
	Mat src = imread("./origin_data/75.png", 0);
	if(!src.data){
		perror("read image filed!\n");
	}
	resize(src, src, Size(128, 128));
	threshold(src, src, 80, 255, THRESH_BINARY);
	src = src < 70;
	medianBlur(src, src, 7);
	src.convertTo(src, CV_32F);
	//imshow("done", src);	
	imwrite("./led_test_data/75.png", src);
	//waitKey(0);
	
	return 0;
}

