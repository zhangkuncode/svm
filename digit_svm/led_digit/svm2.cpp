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

void get_train_data();

/*--------------train data is from origin_data---------------*/

int main(){	
	
	Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
 	svm->setKernel(SVM::LINEAR);// 2
//  svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	get_train_data();
	
	train_data.convertTo(train_data, CV_32F);
	train_classes.convertTo(train_classes, CV_32S);
	
	svm->train(train_data, ROW_SAMPLE, train_classes);	

	Mat src = imread("./my_led/73.jpg", 0);
	if(!src.data){
		perror("read image failed\n");
	}
	test_data = src.reshape(0, 1);
	test_data.convertTo(test_data, CV_32F);
	
	auto r = svm->predict(test_data);
	cout<<"result: "<< r <<endl;
	return 0;
}

void get_train_data(){
	int i = 0;
	char file[255];
	char path[255] = "./origin_data";
	for(i = 0; i <= 9; ++i){
		sprintf(file,"%s/%d1.jpg", path, i);
		Mat src = imread(file, 0);
		resize(src, src, Size(128, 128));
		threshold(src, src, 50, 255, THRESH_BINARY);
		src = src < 50;
		if(!src.data){
			perror("read data_image failed\n");
		}

		src = src.reshape(0, 1);
		src.convertTo(src, CV_32F);

		train_classes.push_back(i); 
		train_data.push_back(src);
	}
	Mat src = imread("./origin_data/72.jpg", 0);
	resize(src, src, Size(128, 128));
	threshold(src, src, 100, 255, THRESH_BINARY);
	src = src < 100;
	src = src.reshape(0, 1);
	src.convertTo(src, CV_32F);
	train_classes.push_back(7); 
	train_data.push_back(src);	
}
