#include <opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main()
{
	
	ifstream  infile;
	infile.open("cal_res1.txt");
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
	double Rl[3], Tl[3], fl[2], cl[2], kl[5];
	double fr[2], cr[2], kr[5], Rr[3] = { 0 }, Tr[3] = { 0 };
	infile >> Rl[0] >> Rl[1] >> Rl[2];
	infile >> Tl[0] >> Tl[1] >> Tl[2];
	infile >> fl[0] >> fl[1];
	infile >> cl[0] >> cl[1];
	infile >> kl[0]>>kl[1]>>kl[2]>>kl[3]>>kl[4];
	infile >> fr[0] >> fr[1];
	infile >> cr[0] >> cr[1];
	for (int i = 0; i < 5; i++)
		infile >> kr[i];
	infile.close();
	Mat intrinsicl = (Mat_<float>(3, 3) << fl[0],0,cl[0],
											0,fl[1],cl[1],
											0,0,1);
	Mat intrinsicr = (Mat_<float>(3, 3) << fr[0], 0, cr[0],
											0, fr[1], cr[1],
											0, 0, 1);
	Mat distcoefl = (Mat_<float>(5, 1) << kl[0], kl[1], kl[2], kl[3], kl[4]);
	Mat distcoefr = (Mat_<float>(5, 1) << kr[0], kr[1], kr[2], kr[3], kr[4]);
	Mat imgl = imread("1L4.bmp", 0);
	Mat imgr = imread("1R4.bmp", 0);
	Mat RL=Mat::zeros(4,3,CV_32FC1);//罗德里格斯计算旋转矩阵
	Mat vecRl = Mat(3, 1, CV_64FC1, Rl);
	Mat vecTl = Mat(3, 1, CV_64FC1, Tl);
	
	Rodrigues(vecRl, RL);
	
	//Rodrigues(vecTl, TL);
	//RL.convertTo(RL, CV_64FC1);
	//TL.convertTo(TL, CV_32FC1);
	Mat rectRl, rectRr, Pl, Pr, Q;//Rl 3x3 rectification transform (rotation matrix) for the first camera.
									//Rr	Output 3x3 rectification transform(rotation matrix) for the second camera.
		//P1	Output 3x4 projection matrix in the new (rectified)coordinate systems for the first camera.
		//P2	Output 3x4 projection matrix in the new (rectified)coordinate systems for the second camera.
		//Q	Output 4×4 disparity - to - depth mapping matrix(see reprojectImageTo3D).
	//计算极线校正时对左右相机的旋转矩阵。
	/*
	立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
	使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
	stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
	左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
	其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]
	Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的时差
	*/

	Rect validROIL, validROIR;//校正会砍掉一部分，砍完了以后的有效区域
	stereoRectify(intrinsicl, distcoefl, intrinsicr, distcoefr, imgl.size(), RL, vecTl, rectRl, rectRr, Pl, Pr, Q, 0,-1,imgl.size(),&validROIL,&validROIR);//cx_1=cx_2 if CV_CALIB_ZERO_DISPARITY is set
	//cout << rectRl<<endl;//CV_CALIB_ZERO_DISPARITY
	//cout << Pl << endl;
	//映射变换计算函数 initUndistortRectifyMap()
	//	该函数功能是计算畸变矫正和立体校正的映射变换。
	//Mat map11, map12, map21, map22;
	//initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	//initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
	Mat maplx, maply, maprx, mapry;//x和y分别有映射函数，maplx是从畸变校正后到畸变校正前的映射，
								//即要算校正后某点灰度值，给定它的坐标，可以返回畸变校正前的坐标值
	initUndistortRectifyMap(intrinsicl, distcoefl, rectRl, Pl, Size(imgl.cols, imgl.rows), CV_32FC1, maplx, maply);
	initUndistortRectifyMap(intrinsicr, distcoefr, rectRr, Pr, Size( imgl.cols, imgl.rows), CV_32FC1, maprx, mapry);
	Mat rectimgl, rectimgr;
	//int newx = 0;
	//newx = imgl.size().width - maplx.at<CV_32FC1>(800.0, 1000.0)[0];
	//cout << newx;
	remap(imgl, rectimgl, maplx, maply, INTER_LINEAR);
	remap(imgr, rectimgr, maprx, mapry, INTER_LINEAR);
	
	rectimgl.convertTo(rectimgl, CV_8UC1);
	//namedWindow("校正后左图", 0);
	//imshow("校正后左图", rectimgl);
	//std::cout <<rectimgl << std::endl;
	//imshow("左图", imgl);
	//namedWindow("校正后右图", 0);
	//imshow("校正后右图", rectimgr);


	Size imageSize = Size(imgl.cols, imgl.rows);
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC1);


	/*左图像画到画布上*/
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));								//得到画布的一部分
	resize(rectimgl, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);		//把图像缩放到跟canvasPart一样大小
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),				//获得被截取的区域	
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);						//画上一个矩形

	cout << "Painted ImageL" << endl;

	/*右图像画到画布上*/
	canvasPart = canvas(Rect(w, 0, w, h));										//获得画布的另一部分
	resize(rectimgr, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	/*画上对应的线条*/
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

	imshow("rectified", canvas);





	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);//cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
									//参数1，允许视差最大值，参数2，视察窗口大小
	Rect roileft, roiright;
	// setter
	//对视差生成效果影响较大的主要参数是setSADWindowSize、setNumberDisparities和setUniquenessRatio，
	//bm->setROI1(roileft);
	//bm->setROI2(roiright);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(-128);//确定匹配搜索从哪里开始  默认值是0
	bm->setNumDisparities(256);//在该数值确定的视差范围内进行搜索,视差窗口	int unitDisparity = 15;int numberOfDisparities = unitDisparity * 16;//40
	bm->setTextureThreshold(10); //保证有足够的纹理以克服噪声
	bm->setUniquenessRatio(10); //使用匹配功能模式
	bm->setSpeckleWindowSize(100);//检查视差连通区域变化度的窗口大小, 值为0时取消 speckle 检查
	bm->setSpeckleRange(32); // 视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
	bm->setPreFilterSize(9);
	
	bm->setBlockSize(21); 
	
	
	//  即最大视差值与最小视差值之差, 大小必须是16的整数倍
	
	

	//// getter
	//int pfs = bm->getPreFilterSize();
	//int pfc = bm->getPreFilterCap();
	//int bs = bm->getBlockSize();
	//int md = bm->getMinDisparity();
	//int nd = bm->getNumDisparities();
	//int tt = bm->getTextureThreshold();
	//int ur = bm->getUniquenessRatio();
	//int sw = bm->getSpeckleWindowSize();
	//int sr = bm->getSpeckleRange();
	Mat dispbm,dispsgbm;
	// Compute disparity
	bm->compute(rectimgl, rectimgr, dispbm);

	int sgbmWinSize = 21;
	cv::Ptr<cv::StereoSGBM>  sgbm = cv::StereoSGBM::create(16, 9);;
	sgbm->setPreFilterCap(70); //preFilterCap：水平sobel预处理后，映射滤波器大小。默认为15
	/*preFilterCap（）匹配图像预处理
		两种立体匹配算法都要先对输入图像做预处理，OpenCV源码中中调用函数 static void prefilterXSobel(const cv::Mat& src, cv::Mat& dst, int preFilterCap)，参数设置中preFilterCap在此函数中用到。函数步骤如下，作用主要有两点：对于无纹理区域，能够排除噪声干扰；对于边界区域，能够提高边界的区分性，利于后续的匹配代价计算：
		先利用水平Sobel算子求输入图像x方向的微分值Value；
		如果Value < -preFilterCap, 则Value = 0;
	如果Value > preFilterCap, 则Value = 2 * preFilterCap;
	如果Value >= -preFilterCap &&Value <= preFilterCap, 则Value = Value + preFilterCap;
	输出处理后的图像作为下一步计算匹配代价的输入图像。*/

	//对视差生成效果影响较大的主要参数是setSADWindowSize、setNumberDisparities和setUniquenessRatio，
	sgbm->setBlockSize(13) ;// SAD窗口大小SADWindowSize:计算代价步骤中SAD窗口的大小。由源码得，此窗口默认大小为5。
	int cn = imgl.channels();
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);//第四、五个参数P1，P2：控制视差图的光滑度
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(-256);//第一个参数minDisparity，一般情况下为0，但有可能矫正算法会移动图像，因此，参数需要进行调整minDisparity：
						//		//最小视差，默认为0。此参数决定左图中的像素点在右图匹配搜索的起点。int 类型
	sgbm->setNumDisparities(512);//第二个参数numDisparities,最大视差减最小视差，现在的算法中，参数必须为16所整除视差搜索范围，其值必须为16的整数倍（CV_Assert( D % 16 == 0 );）。最大搜索边界= numberOfDisparities+ minDisparity。
	sgbm->setUniquenessRatio(10); //唯一性检测参数。对于左图匹配像素点来说，先定义在numberOfDisparities搜索区间内的最低代价为mincost，次低代价为secdmincost。如果满足
		//即说明最低代价和次第代价相差太小，也就是匹配的区分度不够，就认为当前匹配像素点是误匹配的。
	sgbm->setSpeckleWindowSize(80); //视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。
	sgbm->setSpeckleRange(50); //视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。
	sgbm->setDisp12MaxDiff(1);// 左右一致性检测最大容许误差阈值。int 类型
	enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };
	int alg = STEREO_SGBM;
	if (alg == STEREO_HH)
		sgbm->setMode(cv::StereoSGBM::MODE_HH);
	else if (alg == STEREO_SGBM)
		sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
	else if (alg == STEREO_3WAY)
		sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
	sgbm->compute(rectimgl, rectimgr, dispsgbm);

	Mat disp32;
	dispsgbm.convertTo(disp32, CV_32FC1);

	normalize(disp32, disp32, 0, 1, CV_MINMAX);
	disp32 = disp32 * 255;
	disp32.convertTo(disp32, CV_8UC1);
	if ("归一化视差图.bmp")
		imwrite("归一化视差图.jpg",disp32);
	namedWindow("视差图sgbm", 0);
	imshow("视差图sgbm", dispsgbm);



	cout << "wait key" << endl;

	cvWaitKey(0);
	return 0;

	//Mat rectifyLeftX(heightRectified, widthRectified, CV_32FC1);
	//Mat rectifyLeftY(heightRectified, widthRectified, CV_32FC1);

	////载入摄像机内外参数
	//Mat T(3, 1, CV_64FC1);
	//double *dataT = (double *)T.data;
	//double fc[2];
	//double cc[2];
	//fscanf(fp, "%*lf%*lf%*lf%lf%lf%lf", dataT, dataT + 1, dataT + 2);

}