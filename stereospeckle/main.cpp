#include <opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main()
{
	
	ifstream  infile;
	infile.open("cal_res1.txt");
	assert(infile.is_open());   //��ʧ��,�����������Ϣ,����ֹ�������� 
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
	Mat RL=Mat::zeros(4,3,CV_32FC1);//�޵����˹������ת����
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
		//Q	Output 4��4 disparity - to - depth mapping matrix(see reprojectImageTo3D).
	//���㼫��У��ʱ�������������ת����
	/*
	����У����ʱ����Ҫ����ͼ���沢���ж�׼ ��ʹ������ƥ����ӵĿɿ�
	ʹ������ͼ����ķ������ǰ���������ͷ��ͼ��ͶӰ��һ�������������ϣ�����ÿ��ͼ��ӱ�ͼ��ƽ��ͶӰ������ͼ��ƽ�涼��Ҫһ����ת����R
	stereoRectify �����������ľ��Ǵ�ͼ��ƽ��ͶӰ����������ƽ�����ת����Rl,Rr�� Rl,Rr��Ϊ�������ƽ���ж�׼��У����ת����
	���������Rl��ת�����������Rr��ת֮������ͼ����Ѿ����沢���ж�׼�ˡ�
	����Pl,PrΪ���������ͶӰ�����������ǽ�3D�������ת����ͼ���2D�������:P*[X Y Z 1]' =[x y w]
	Q����Ϊ��ͶӰ���󣬼�����Q���԰�2άƽ��(ͼ��ƽ��)�ϵĵ�ͶӰ��3ά�ռ�ĵ�:Q*[x y d 1] = [X Y Z W]������dΪ��������ͼ���ʱ��
	*/

	Rect validROIL, validROIR;//У���ῳ��һ���֣��������Ժ����Ч����
	stereoRectify(intrinsicl, distcoefl, intrinsicr, distcoefr, imgl.size(), RL, vecTl, rectRl, rectRr, Pl, Pr, Q, 0,-1,imgl.size(),&validROIL,&validROIR);//cx_1=cx_2 if CV_CALIB_ZERO_DISPARITY is set
	//cout << rectRl<<endl;//CV_CALIB_ZERO_DISPARITY
	//cout << Pl << endl;
	//ӳ��任���㺯�� initUndistortRectifyMap()
	//	�ú��������Ǽ���������������У����ӳ��任��
	//Mat map11, map12, map21, map22;
	//initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	//initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
	Mat maplx, maply, maprx, mapry;//x��y�ֱ���ӳ�亯����maplx�Ǵӻ���У���󵽻���У��ǰ��ӳ�䣬
								//��Ҫ��У����ĳ��Ҷ�ֵ�������������꣬���Է��ػ���У��ǰ������ֵ
	initUndistortRectifyMap(intrinsicl, distcoefl, rectRl, Pl, Size(imgl.cols, imgl.rows), CV_32FC1, maplx, maply);
	initUndistortRectifyMap(intrinsicr, distcoefr, rectRr, Pr, Size( imgl.cols, imgl.rows), CV_32FC1, maprx, mapry);
	Mat rectimgl, rectimgr;
	//int newx = 0;
	//newx = imgl.size().width - maplx.at<CV_32FC1>(800.0, 1000.0)[0];
	//cout << newx;
	remap(imgl, rectimgl, maplx, maply, INTER_LINEAR);
	remap(imgr, rectimgr, maprx, mapry, INTER_LINEAR);
	
	rectimgl.convertTo(rectimgl, CV_8UC1);
	//namedWindow("У������ͼ", 0);
	//imshow("У������ͼ", rectimgl);
	//std::cout <<rectimgl << std::endl;
	//imshow("��ͼ", imgl);
	//namedWindow("У������ͼ", 0);
	//imshow("У������ͼ", rectimgr);


	Size imageSize = Size(imgl.cols, imgl.rows);
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC1);


	/*��ͼ�񻭵�������*/
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));								//�õ�������һ����
	resize(rectimgl, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);		//��ͼ�����ŵ���canvasPartһ����С
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),				//��ñ���ȡ������	
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);						//����һ������

	cout << "Painted ImageL" << endl;

	/*��ͼ�񻭵�������*/
	canvasPart = canvas(Rect(w, 0, w, h));										//��û�������һ����
	resize(rectimgr, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	/*���϶�Ӧ������*/
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

	imshow("rectified", canvas);





	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);//cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
									//����1�������Ӳ����ֵ������2���Ӳ촰�ڴ�С
	Rect roileft, roiright;
	// setter
	//���Ӳ�����Ч��Ӱ��ϴ����Ҫ������setSADWindowSize��setNumberDisparities��setUniquenessRatio��
	//bm->setROI1(roileft);
	//bm->setROI2(roiright);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(-128);//ȷ��ƥ�����������￪ʼ  Ĭ��ֵ��0
	bm->setNumDisparities(256);//�ڸ���ֵȷ�����ӲΧ�ڽ�������,�Ӳ��	int unitDisparity = 15;int numberOfDisparities = unitDisparity * 16;//40
	bm->setTextureThreshold(10); //��֤���㹻�������Կ˷�����
	bm->setUniquenessRatio(10); //ʹ��ƥ�书��ģʽ
	bm->setSpeckleWindowSize(100);//����Ӳ���ͨ����仯�ȵĴ��ڴ�С, ֵΪ0ʱȡ�� speckle ���
	bm->setSpeckleRange(32); // �Ӳ�仯��ֵ�����������Ӳ�仯������ֵʱ���ô����ڵ��Ӳ�����
	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
	bm->setPreFilterSize(9);
	
	bm->setBlockSize(21); 
	
	
	//  ������Ӳ�ֵ����С�Ӳ�ֵ֮��, ��С������16��������
	
	

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
	sgbm->setPreFilterCap(70); //preFilterCap��ˮƽsobelԤ�����ӳ���˲�����С��Ĭ��Ϊ15
	/*preFilterCap����ƥ��ͼ��Ԥ����
		��������ƥ���㷨��Ҫ�ȶ�����ͼ����Ԥ����OpenCVԴ�����е��ú��� static void prefilterXSobel(const cv::Mat& src, cv::Mat& dst, int preFilterCap)������������preFilterCap�ڴ˺������õ��������������£�������Ҫ�����㣺���������������ܹ��ų��������ţ����ڱ߽������ܹ���߽߱�������ԣ����ں�����ƥ����ۼ��㣺
		������ˮƽSobel����������ͼ��x�����΢��ֵValue��
		���Value < -preFilterCap, ��Value = 0;
	���Value > preFilterCap, ��Value = 2 * preFilterCap;
	���Value >= -preFilterCap &&Value <= preFilterCap, ��Value = Value + preFilterCap;
	���������ͼ����Ϊ��һ������ƥ����۵�����ͼ��*/

	//���Ӳ�����Ч��Ӱ��ϴ����Ҫ������setSADWindowSize��setNumberDisparities��setUniquenessRatio��
	sgbm->setBlockSize(13) ;// SAD���ڴ�СSADWindowSize:������۲�����SAD���ڵĴ�С����Դ��ã��˴���Ĭ�ϴ�СΪ5��
	int cn = imgl.channels();
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);//���ġ��������P1��P2�������Ӳ�ͼ�Ĺ⻬��
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(-256);//��һ������minDisparity��һ�������Ϊ0�����п��ܽ����㷨���ƶ�ͼ����ˣ�������Ҫ���е���minDisparity��
						//		//��С�ӲĬ��Ϊ0���˲���������ͼ�е����ص�����ͼƥ����������㡣int ����
	sgbm->setNumDisparities(512);//�ڶ�������numDisparities,����Ӳ����С�Ӳ���ڵ��㷨�У���������Ϊ16�������Ӳ�������Χ����ֵ����Ϊ16����������CV_Assert( D % 16 == 0 );������������߽�= numberOfDisparities+ minDisparity��
	sgbm->setUniquenessRatio(10); //Ψһ�Լ�������������ͼƥ�����ص���˵���ȶ�����numberOfDisparities���������ڵ���ʹ���Ϊmincost���εʹ���Ϊsecdmincost���������
		//��˵����ʹ��ۺʹεڴ������̫С��Ҳ����ƥ������ֶȲ���������Ϊ��ǰƥ�����ص�����ƥ��ġ�
	sgbm->setSpeckleWindowSize(80); //�Ӳ���ͨ�������ص�����Ĵ�С������ÿһ���Ӳ�㣬������ͨ��������ص����С��speckleWindowSizeʱ����Ϊ���Ӳ�ֵ��Ч������㡣
	sgbm->setSpeckleRange(50); //�Ӳ���ͨ�������ڼ���һ���Ӳ�����ͨ����ʱ������һ�����ص��Ӳ�仯����ֵ����speckleRange����Ϊ��һ���Ӳ����ص�͵�ǰ�Ӳ����ص��ǲ���ͨ�ġ�
	sgbm->setDisp12MaxDiff(1);// ����һ���Լ��������������ֵ��int ����
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
	if ("��һ���Ӳ�ͼ.bmp")
		imwrite("��һ���Ӳ�ͼ.jpg",disp32);
	namedWindow("�Ӳ�ͼsgbm", 0);
	imshow("�Ӳ�ͼsgbm", dispsgbm);



	cout << "wait key" << endl;

	cvWaitKey(0);
	return 0;

	//Mat rectifyLeftX(heightRectified, widthRectified, CV_32FC1);
	//Mat rectifyLeftY(heightRectified, widthRectified, CV_32FC1);

	////����������������
	//Mat T(3, 1, CV_64FC1);
	//double *dataT = (double *)T.data;
	//double fc[2];
	//double cc[2];
	//fscanf(fp, "%*lf%*lf%*lf%lf%lf%lf", dataT, dataT + 1, dataT + 2);

}