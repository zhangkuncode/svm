#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;

using namespace cv;
using namespace cv::ml;

Mat train_data;
Mat train_classes;
Mat test_data;
const int K = 3;

void get_train_data();
int main(){	
	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(K);
	knn->setIsClassifier(true);

	get_train_data();
	
	train_data.convertTo(train_data, CV_32F);
	train_classes.convertTo(train_classes, CV_32S);
	
	knn->train(train_data, ROW_SAMPLE, train_classes);

	Mat src = imread("./led_test_data/75.png", 0);
	if(!src.data){
		perror("read image failed\n");
	}
	test_data = src.reshape(0, 1);
	test_data.convertTo(test_data, CV_32F);
	
	auto r = knn->findNearest(test_data, K, noArray());
	cout<<"result: "<< r <<endl;
	return 0;
}

void get_train_data(){
	char path[255] = "./my_led_data";
	for(int i = 0; i <= 9; ++i){
		for(int j = 1; j <= 19; ++j){
			char file[255];
			if(j < 10){
				sprintf(file, "%s/%d/0%d.jpg", path, i, j);
				Mat temp = imread(file, 0);
				temp = temp.reshape(0, 1);
				temp.convertTo(temp, CV_32F);
				train_data.push_back(temp);
				train_classes.push_back(i);
			} else if(j >= 10) {
				sprintf(file, "%s/%d/%d.jpg", path, i, j);
				Mat temp = imread(file, 0);
				temp = temp.reshape(0, 1);
				temp.convertTo(temp, CV_32F);
				train_data.push_back(temp);
				train_classes.push_back(i);
			}
		}
	}
}
