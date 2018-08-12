#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat element3 = getStructuringElement(MORPH_RECT, Size(7, 7));

Mat element4 = getStructuringElement(MORPH_CROSS, Size(3, 3));
Mat element5 = getStructuringElement(MORPH_CROSS, Size(5, 5));
Mat element6 = getStructuringElement(MORPH_CROSS, Size(7, 7));
Mat element7 = getStructuringElement(MORPH_CROSS, Size(9, 9));

Mat element8 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
Mat element9 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
Mat element10 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
Mat element11 = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));

int main(){
	Mat src = imread("./my_led/91.jpg");

	Mat out1;
	dilate(src, out1, element1);
	imwrite("./my_led2/9/02.jpg", out1);
	Mat out2;
	dilate(src, out2, element2);
	imwrite("./my_led2/9/03.jpg", out2);
	Mat out3;
	dilate(src, out3, element3);
	imwrite("./my_led2/9/04.jpg", out3);
	Mat out4;
	dilate(src, out4, element3);
	imwrite("./my_led2/9/05.jpg", out4);
	Mat out5;
	dilate(src, out5, element5);
	imwrite("./my_led2/9/06.jpg", out5);
	Mat out6;
	dilate(src, out6, element6);
	imwrite("./my_led2/9/07.jpg", out6);
	Mat out7;
	dilate(src, out7, element7);
	imwrite("./my_led2/9/08.jpg", out7);
	Mat out8;
	dilate(src, out8, element8);
	imwrite("./my_led2/9/09.jpg", out8);
	Mat out9;
	dilate(src, out9, element9);
	imwrite("./my_led2/9/10.jpg",out9);
	Mat out10;
	dilate(src, out10, element10);
	imwrite("./my_led2/9/11.jpg", out10);
	Mat out11;
	dilate(src, out11, element11);
	imwrite("./my_led2/9/12.jpg", out11);

	return 0;
}
