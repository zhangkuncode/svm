#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

int main(){
    int width = 600, height = 600;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    float trainingData[7][2] = {{100, 100}, {100, 200}, {200, 100}, {300, 300}, 
		                        {300, 200}, {400, 200}, {500, 300} };
	
    int labels[7] = {1, 1, 1, -1, -1, -1, 1};
    // the type of image is stable
	Mat trainingDataMat(7, 2, CV_32FC1, trainingData);
    Mat labelsMat(7, 1, CV_32SC1, labels);

    // 训练 SVM
    /* 如果只是简单的点分类，svm的参数设置就这两行 OK
	   但如果是其它更为复杂的分类，则需要设置更多的参数*/
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);

	svm->setKernel(SVM::LINEAR);
	/*[opencv]defining termination criteria for iterative algorithms*/
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->setC(0.0000001);    

	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);


    // 显示二分分类的结果
    Vec3b green(0, 255, 0), blue(255, 0, 0);
    for (int i = 0; i < image.rows; ++i){
        for (int j = 0; j < image.cols; ++j){
            Mat sampleMat = (Mat_<float>(1, 2) << j, i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i, j) = blue;
            else if (response == -1)
                image.at<Vec3b>(i, j) = green;
        }
	}

    // 画出训练样本数据
    int thickness = -1;// whether is filed     -1:yes   >0:width of lines
    int lineType = 8;
	//thickness 如果是正数，表示组成圆的线条的粗细程度。否则，表示圆是否被填充
	//line_type 线条的类型。默认是8
	/*                         radius                           */
	circle(image, Point(100,100), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(100,200), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(200,100), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(300,300), 5, Scalar(255, 225, 255), thickness, lineType);
    circle(image, Point(300,200), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(400,200), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(500,300), 5, Scalar(255, 255, 255), thickness, lineType);
    
    // 显示出支持向量
    thickness = 2;
    lineType = 8;
    Mat sv = svm->getUncompressedSupportVectors();
    //Mat sv = svm->getSupportVectors();// that is not OK
    for (int i = 0; i < sv.rows; ++i){
        const float* v = sv.ptr<float>(i);
        circle(image, Point((int)v[0], (int)v[1]), 15, 
			   Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("regularize.png", image);        // 保存训练的结果
  
	// the window can change size
	namedWindow("SVM Simple Example", CV_WINDOW_NORMAL);
	imshow("SVM Simple Example", image); 
    
	waitKey(0);
}
