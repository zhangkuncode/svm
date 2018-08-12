#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;

using namespace cv;
using namespace cv::ml;

int samples = 100;
int classes = 10;
char file_path[255] = "../English/Fnt/Sample0";
char file_path2[225] = "img0";
char file_path3[225] = "-0";
Mat train_data;
Mat train_classes;
Mat test_data;

void get_train_data();
void set_test_data(const char* c);

int main(){	
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    
	clock_t start, end;
	start = clock();
	get_train_data();	
	end = clock();

	train_data.convertTo(train_data, CV_32F);
	train_classes.convertTo(train_classes, CV_32S);
	
	svm->train(train_data, ROW_SAMPLE, train_classes);	
	
	int i = 0, j = 0, error = 0;
	char file[255];
	for(i = 1; i <= 10; ++i){
		for(j = 509; j <= 609; ++j){
			if(i < 10){
				if(j < 10){
					sprintf(file,"%s0%d/%s0%d%s000%d.png",
							file_path, i, file_path2, i, file_path3, j);
				} else if(j < 100) {
					sprintf(file,"%s0%d/%s0%d%s00%d.png",
							file_path, i, file_path2, i, file_path3, j);
				} else if(j < 1000){
					sprintf(file,"%s0%d/%s0%d%s0%d.png",
							file_path, i, file_path2, i, file_path3, j);
				} else {
					sprintf(file,"%s0%d/%s0%d%s%d.png",
							file_path, i, file_path2, i, file_path3, j);
				}
			} else {
				if(j < 10){
					sprintf(file,"%s10/%s%d%s000%d.png",
							file_path, file_path2, i, file_path3, j);
				} else if(j < 100) {
					sprintf(file,"%s10/%s%d%s00%d.png",
							file_path, file_path2, i, file_path3, j);
				} else if(j < 1000){
					sprintf(file,"%s10/%s%d%s0%d.png",
							file_path, file_path2, i, file_path3, j);
				} else {
					sprintf(file,"%s10/%s%d%s%d.png",
							file_path, file_path2, i, file_path3, j);
				}
			} 

			Mat temp = imread(file, 0);
			if(!temp.data){
				perror("read file failed");				
			}
			temp = temp.reshape(0, 1);
			Mat mytest;
			mytest.push_back(temp);
			mytest.convertTo(mytest, CV_32F);
			auto r = svm->predict(mytest);
			if(r != (i - 1)){
				error++;
			}
		}
	}
	cout << "cost time: " << (double)(end - start) / CLOCKS_PER_SEC
		 << " s" << endl;
	cout << "accuracy: " << ( (float)(1000 - error) / 10 ) << "%" << endl;
	
	return 0;
}

void get_train_data(){
	int i = 0, j = 0;
	char file[255];
	for(i = 1; i <= classes; ++i){
		for(j = 1; j <= samples; ++j){
			if(i < 10){
				if(j < 10){
					sprintf(file,"%s0%d/%s0%d%s000%d.png",
							file_path, i, file_path2, i, file_path3, j);
				} else if(j < 100) {
					sprintf(file,"%s0%d/%s0%d%s00%d.png",
							file_path, i, file_path2, i, file_path3, j);
				} else if(j < 1000){
					sprintf(file,"%s0%d/%s0%d%s0%d.png",
							file_path, i, file_path2, i, file_path3, j);
				} else {
					sprintf(file,"%s0%d/%s0%d%s%d.png",
							file_path, i, file_path2, i, file_path3, j);
				}
			} else {
				if(j < 10){
					sprintf(file,"%s10/%s%d%s000%d.png",
							file_path, file_path2, i, file_path3, j);
				} else if(j < 100) {
					sprintf(file,"%s10/%s%d%s00%d.png",
							file_path, file_path2, i, file_path3, j);
				} else if(j < 1000){
					sprintf(file,"%s10/%s%d%s0%d.png",
							file_path, file_path2, i, file_path3, j);
				} else {
					sprintf(file,"%s10/%s%d%s%d.png",
							file_path, file_path2, i, file_path3, j);
				}
			} 

			Mat src = imread(file, 0);		
			if(!src.data){
				perror("read data_image failed\n");
			}
			src = src.reshape(0, 1);
			src.convertTo(src, CV_32F);

			train_classes.push_back(i - 1); 
			train_data.push_back(src);
		}
	}
}

void set_test_data(const char* c){
	Mat temp = imread(c, 0);
	if(!temp.data){
		perror("read test_image failed\n");
	}
		
	temp = temp.reshape(0, 1);
	test_data.push_back(temp);	
	test_data.convertTo(test_data, CV_32F);
}
