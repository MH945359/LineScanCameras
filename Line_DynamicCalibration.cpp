#include"Static_Calibration.h"
using namespace std;
using namespace Eigen;
//struct MFunctor
//{
//
//	double U_guan, V_guan, Xw_guan, Yw_guan, Zw_guan;
//
//	//构造函数，用已知的x、y数据对其赋值
//	 //CostFunctor(double xc, double u)
//	MFunctor(double u, double v, double xw, double yw, double zw)
//	{
//		U_guan = u;
//		V_guan = v;
//		Xw_guan = xw;
//		Yw_guan = yw;
//		Zw_guan = zw;
//	}
//	//重载括号运算符，两个参数分别是估计的参数和由该参数计算得到的残差
//	template <typename T>
//	bool operator()(const T* const params, T* residual)const
//	{
//		residual[0] = (params[0] * T(Xw_guan) + params[1] * T(Yw_guan) + params[2] * T(Zw_guan) + params[3]) / (params[8] * T(Xw_guan) + params[9] * T(Yw_guan) + params[10] * T(Zw_guan) + 1.0) - T(U_guan);
//		residual[1] = (params[4] * T(Xw_guan) + params[5] * T(Yw_guan) + params[6] * T(Zw_guan) + params[7]) - T(V_guan);
//
//		return true;
//		return true;
//	}
//};

struct MFunctor
{

	double U_guan, V_guan, Xw_guan, Yw_guan, Zw_guan,k11_guan,k12_guan,k13_guan,k22_guan,k32_guan, tx_guan, tz_guan;

	//构造函数，用已知的x、y数据对其赋值
	 //CostFunctor(double xc, double u)
	MFunctor(cv::Point2d &pix_point, cv::Point3d & WPoint, Eigen::Matrix<double, 3, 3>&M, Eigen::Matrix2f& M1)
	{
		U_guan = pix_point.x;
		V_guan = pix_point.y;
		Xw_guan = WPoint.x;
		Yw_guan = WPoint.y;
		Zw_guan = WPoint.z;
		k11_guan = M(0, 0);
		k12_guan = M(0, 1);
		k13_guan = M(0, 2);
		k22_guan = M(1, 1);
		k32_guan = M(2, 1);
		tx_guan = M1(0, 1);
		tz_guan = M1(1, 1);
	}
	/*	(ceres::cos(params[0]) * ceres::cos(params[1])),
		(ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) - ceres::sin(params[0]) * ceres::cos(params[2])),
		(ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) + ceres::cos(params[0]) * ceres::sin(params[2])),
		((ceres::cos(params[0]) * ceres::cos(params[1])) * Cam_Poses[0](0, 1)+ (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) - ceres::sin(params[0]) * ceres::cos(params[2])) * params[3] + (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) + ceres::cos(params[0]) * ceres::sin(params[2])) * Cam_Poses[0](1, 1)),
		
		(ceres::sin(params[0]) * ceres::cos(params[1])),
		(ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) + ceres::cos(params[0]) * ceres::cos(params[2])),
		(ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])),
		((ceres::sin(params[0]) * ceres::cos(params[1])) * Cam_Poses[0](0, 1) + (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) 
			+ ceres::cos(params[0]) * ceres::cos(params[2])) * params[3] 
			+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])) * Cam_Poses[0](1, 1)),
	    
		(-ceres::sin(params[1])),
		(ceres::cos(params[1]) * ceres::sin(params[2])),
		(ceres::cos(params[1]) * ceres::cos(params[2])),
		((-ceres::sin(params[1])) * Cam_Poses[0](0, 1) + (ceres::cos(params[1]) * ceres::sin(params[2])) * params[3] + (ceres::cos(params[1]) * ceres::cos(params[2])) * Cam_Poses[0](1, 1));*/

	//重载括号运算符，两个参数分别是估计的参数和由该参数计算得到的残差
	template <typename T>
	bool operator()(const T* const params, T* residual)const
	{

		residual[0] = (T(k32_guan) * ((ceres::sin(params[0]) * ceres::cos(params[1])) * T(Xw_guan) 
			+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) + ceres::cos(params[0]) * ceres::cos(params[2])) * T(Yw_guan) 
			+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])) * T(Zw_guan)
			+ ((ceres::sin(params[0]) * ceres::cos(params[1])) * T(tx_guan) + (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2])
			+ ceres::cos(params[0]) * ceres::cos(params[2])) * params[3]+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])) * T(tz_guan))) + (-ceres::sin(params[1])) * T(Xw_guan)
			+ (ceres::cos(params[1]) * ceres::sin(params[2])) * T(Yw_guan) + (ceres::cos(params[1]) * ceres::cos(params[2])) * T(Zw_guan) + ((-ceres::sin(params[1])) * T(tx_guan) + (ceres::cos(params[1]) * ceres::sin(params[2])) * params[3]
			+ (ceres::cos(params[1]) * ceres::cos(params[2])) * T(tz_guan))) - params[4];
		residual[1] = ((T(k11_guan) * (ceres::cos(params[0]) * ceres::cos(params[1])) + T(k12_guan) * (ceres::sin(params[0]) * ceres::cos(params[1])) + T(k13_guan) * (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) 
			+ ceres::cos(params[0]) * ceres::cos(params[2]))) * T(Xw_guan)+ (T(k11_guan) * (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) - ceres::sin(params[0]) * ceres::cos(params[2])) + T(k12_guan) * (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) + ceres::cos(params[0]) * ceres::cos(params[2]))
			+ T(k13_guan) * (ceres::cos(params[1]) * ceres::sin(params[2]))) * T(Yw_guan)+ (T(k11_guan) * (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) + ceres::cos(params[0]) * ceres::sin(params[2])) + T(k12_guan) * (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2]))
			+ T(k13_guan) * (ceres::cos(params[1]) * ceres::cos(params[2]))) * T(Zw_guan)+ (T(k11_guan) * ((ceres::sin(params[0]) * ceres::cos(params[1])) * T(tx_guan) + (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2])
			+ ceres::cos(params[0]) * ceres::cos(params[2])) * params[3]+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])) * T(tz_guan)) + T(k12_guan) * ((ceres::sin(params[0]) * ceres::cos(params[1])) * T(tx_guan) + (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2])
			+ ceres::cos(params[0]) * ceres::cos(params[2])) * params[3]+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])) * T(tz_guan)) + T(k13_guan) * ((-ceres::sin(params[1])) * T(tx_guan)
			+ (ceres::cos(params[1]) * ceres::sin(params[2])) * params[3] + (ceres::cos(params[1]) * ceres::cos(params[2])) * T(tz_guan)))) - params[4] * T(U_guan);
		residual[2] = T(k22_guan) * ((ceres::sin(params[0]) * ceres::cos(params[1])) * T(Xw_guan) + (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) + ceres::cos(params[0]) * ceres::cos(params[2])) * T(Yw_guan)
			+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])) * T(Zw_guan) + params[3]) - T(V_guan);
		residual[3] = (ceres::cos(params[0]) * ceres::cos(params[1])) * (ceres::sin(params[0]) * ceres::cos(params[1]))
			+ (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) - ceres::sin(params[0]) * ceres::cos(params[2])) * (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) + ceres::cos(params[0]) * ceres::cos(params[2]))
			+ (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) + ceres::cos(params[0]) * ceres::sin(params[2])) * (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2]));

	/*	residual[0] = (T(k32_guan) * (params[4] * T(Xw_guan) + params[5] * T(Yw_guan) + params[6] * T(Zw_guan) + params[7]) + params[8] * T(Xw_guan) + params[9] * T(Yw_guan) + params[10] * T(Zw_guan) + params[11]) - params[12];
		residual[1] = ((T(k11_guan) * params[0] + T(k12_guan) * params[4] + T(k13_guan) * params[5]) * T(Xw_guan)
			+ (T(k11_guan) * params[1] + T(k12_guan) * params[5] + T(k13_guan) * params[9]) * T(Yw_guan)
			+ (T(k11_guan) * params[2] + T(k12_guan) * params[6] + T(k13_guan) * params[10]) * T(Zw_guan)
			+ (T(k11_guan) * params[3] + T(k12_guan) * params[7] + T(k13_guan) * params[11])) - params[12] * T(U_guan);
		residual[2] = params[4] * T(k22_guan) * T(Xw_guan) + params[5] * T(k22_guan) * T(Yw_guan) + params[6] * T(k22_guan) * T(Zw_guan) + params[7] * T(k22_guan) - T(V_guan);*
	residual[3] = params[0] * params[2] + params[4] * params[6] + params[8] * params[10];
		residual[3] = params[0] * params[0] + params[4] * params[4] + params[8] * params[8]-1.0;
		residual[4] = params[1] * params[1] + params[5] * params[5] + params[9] * params[9] - 1.0;
		residual[5] = params[2] * params[2] + params[6] * params[6] + params[10] * params[10] - 1.0;
		residual[4] = params[0] * params[1] + params[4] * params[5] + params[8] * params[9];*/
		return true;
	}
};
int main()
{
	//cv::Mat img = cv::imread(R"(F:\Line_Picture\24.1.8Line\3.bmp)", cv::IMREAD_GRAYSCALE);
	//cv::Mat img = cv::imread(R"(D:\test_pictures\Line_Picture\24.1.24\2.bmp)", cv::IMREAD_GRAYSCALE);
    //cv::Mat img = cv::imread(R"(D:\test_pictures\Line_Picture\24.1.15_linepicture\2.bmp)", cv::IMREAD_GRAYSCALE);
	//cv::Mat img = cv::imread(R"(F:\Line_Picture\24.1.22\11.bmp)", cv::IMREAD_GRAYSCALE);
	cv::Mat img = cv::imread(R"(D:\test_pictures\Line_Picture\24.2.26\1.bmp)", cv::IMREAD_GRAYSCALE);
	//获取像素坐标系角点坐标（x,y）
	//选择行数复制500行生成图片
	/*for (int z = 0; z < 11; z++)
	{*/
		
		int linenum =5;
		int coline = 705;
		cout <<"coline: " << coline << endl;
		//int linenum = 0;
		//cout << "Select Line Number: ";//提取几条线
		//cin >> linenum;
		//int coline = 0;
		//cout << "Select Row: ";//提取第多少行的像素
	 //    cin >> coline;
		int jiange = 3.6/0.026;
		vector<cv::Mat>imgsRows;//存放提取的图像
		vector<int>colines;
		for (int i = 0; i < linenum; i++)
		{
			cv::Mat selectedRows;
			cv::Mat Rows;
			selectedRows = img(cv::Range(coline, coline + 1), cv::Range::all());
			cv::repeat(selectedRows, 500, 1, Rows);
			imgsRows.push_back(Rows);
			colines.push_back(coline);
			cv::line(img, cv::Point(0, coline), cv::Point(img.cols, coline), cv::Scalar(0, 0, 255), 1, 8);
			coline += jiange;
		}

		//自适应阈值提取像素坐标系角点只要竖边的像素的像素点
		vector<vector<cv::Point2d>>pix_points;//图像坐标系坐标点
		for (int i = 0; i < linenum; i++)
		{
			cv::Mat binaryImage;
			vector<cv::Point2d>pix_point;
			cv::adaptiveThreshold(imgsRows[i], binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 781, 0);
			cv::Mat labels, centroids, stats;
			int number = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids, 4, CV_32S);
			AreaSelect(number, stats, centroids, pix_point, colines[i]);
			pix_points.push_back(pix_point);
			if (pix_point.size() != 40)
			{
				std::cout << "No." << i + 1 << "张：" << "Pix Points Number Error" << pix_point.size() << endl;
				return 0;
			}
			
		}
		//获取世界坐标点
		vector<double>VerPoint;//竖线的世界坐标点横向x
		double Ver = 0.0;
		for (int i = 0; i < pix_points[0].size() / 2; i++)
		{
			VerPoint.push_back(Ver);
			VerPoint.push_back(0);
			Ver += 8.0;
		}
		//交比不变性
		vector<vector<double>>HypPointcols;
		vector<double>HypPointcol;//斜线的世界坐标点横向x
		for (int i = 0; i < linenum; i++)
		{
			for (int j = 0; j < pix_points[0].size() - 4; j++)
			{
				if (j % 2 == 0)
				{
					double A = pix_points[i][j].x - pix_points[i][j + 2].x;//a(j)-c(j+2)
					double B = pix_points[i][j + 1].x - pix_points[i][j + 4].x;//b(j+1)-d(j+4)
					double C = pix_points[i][j + 1].x - pix_points[i][j + 2].x;//b(j+1)-c(j+2)
					double D = pix_points[i][j].x - pix_points[i][j + 4].x;//a(j)-d(j+4)
					double Cr = A * B / (C * D);
					double p = (VerPoint[j + 2] - VerPoint[j]) / (VerPoint[j + 4] - VerPoint[j]);
					double Hyp = (Cr * VerPoint[j + 2] - p * VerPoint[j + 4]) / (Cr - p);
					HypPointcol.push_back(Hyp);
				}
			}
			HypPointcols.push_back(HypPointcol);
			HypPointcol.clear();
		}
		vector<vector<cv::Point2d>>HypPoints;//斜边的的世界坐标点（x,y）
		vector<cv::Point2d>HypPoint;
		for (int i = 0; i < linenum; i++)
		{
			for (int j = 0; j < HypPointcols[0].size(); j++)
			{
				cv::Point2d HyPoint;
				HyPoint.x = HypPointcols[i][j];
				HyPoint.y = -5.0 * (HypPointcols[i][j] - j * 8.0) + 40;
				HypPoint.push_back(HyPoint);
			}
			HypPoints.push_back(HypPoint);
			HypPoint.clear();
		}
		//得到世界坐标点角点坐标（x,y）
		vector <vector < cv::Point3d >> World_Points;//世界坐标系角点三维坐标
		vector < cv::Point3d > World_Point;
		vector<double>Intercepts;//拟合线截距
		vector<float>rads;
		vector<float>ak;
		for (int i = 0; i < linenum; i++)
		{
			cv::Vec4f lineParams;
			cv::fitLine(HypPoints[i], lineParams, cv::DIST_L2, 0, 0.001, 0.001);//通过最小二乘法拟合斜边点的方程
			float k = lineParams[1] / lineParams[0];
			double b = lineParams[3] - k * lineParams[2];
			Intercepts.push_back(b);
			ak.push_back(k);
			float rad = std::atan(k);
			rads.push_back(rad);
			int a = 0;
			for (int j = 0; j < VerPoint.size() - 4; j++)
			{
				if (j % 2 == 0)
				{
					//竖边世界坐标点
					cv::Point3d val;
					val.y = k * VerPoint[j] + b - Intercepts[0];
					//double ssss = k * HypPointcols[0][1] + b;
					//cout << ssss << endl;
					val.x = VerPoint[j];
					val.z = val.y * std::cos(45 * CV_PI / 180);
					World_Point.push_back(val);
					a++;
				}
				  else
				  {
					  cv::Point3d val1;
					  val1.y = HypPoints[i][a-1].y - Intercepts[0];
					  val1.x = HypPoints[i][a-1].x;
					  val1.z = val1.y * std::cos(45 * CV_PI / 180);
					  World_Point.push_back(val1);
				  }
			}
			World_Points.push_back(World_Point);
			World_Point.clear();
		}
		vector <vector < cv::Point3d >> New_Points;
		vector < cv::Point3d >New_Point;
		for (int i = 0; i < World_Points[0].size(); i++)
		{
			cv::Point3d val;
			val.y = 0;
			val.x = World_Points[0][i].x / std::cos(rads[0]);
			val.z = 0;
			New_Point.push_back(val);
		}
		New_Points.push_back(New_Point);
		New_Point.clear();
		for (int i = 0; i < World_Points.size() - 1; ++i)
		{
			for (int j = 0; j < World_Points[0].size(); j++)
			{
				cv::Point3d val1;
				val1.y = (World_Points[i + 1][j].y - World_Points[0][j].y) * std::cos(rads[i]) * std::cos(45 * CV_PI / 180);
				val1.x = World_Points[i + 1][j].x / std::cos(rads[i + 1]) + val1.y * std::tan(rads[i]);
				val1.z = val1.y * std::tan(45 * CV_PI / 180);
				New_Point.push_back(val1);
			}

			New_Points.push_back(New_Point);
			New_Point.clear();
		}
		vector<double>disssss;
		
		for (int i = 0; i < New_Points.size() - 1; i++)
		{
			double distanssss = 0;
			for (int j = 0; j < New_Points[0].size(); j++)
			{
				if (j % 2 == 1)
				{
					double distance = New_Points[i][j].x-New_Points[i+1][j].x-abs(New_Points[i][j].z - New_Points[i + 1][j].z)/8/std::cos(45 * CV_PI / 180);
					distanssss += distance;
				}
			}
			double sss = distanssss /New_Points[0].size();
			disssss.push_back(sss);

		}
		//静态标定
		vector<vector<double>>Static_World_Points; //静态标定的世界坐标点
		CalibrationConver(linenum, VerPoint, HypPoints, Static_World_Points);//动态标定板坐标转换到静态标定的世界坐标点

		//求解Vy Vx
		double Vyvars = 0;
		double Ydistances = 0;//y方向的差值
		for (int i = 0; i < linenum - 1; i++)
		{
			double ysum = 0;
			double xsum = 0;
			for (int j = 0; j < World_Points[0].size(); j++)
			{
				double yval = World_Points[i + 1][j].y - World_Points[i][j].y;
				double xval = yval * sin(rads[0]);
				ysum += yval;
			}
			double YaverDis = ysum / World_Points[0].size();
			//double Vyvar = YaverDis / (100 * std::cos(rads[0]));
			double Vyvar = YaverDis * std::cos(45 * CV_PI / 180) / (jiange * std::cos(rads[0]));
			Vyvars += Vyvar;//Vy总值
			Ydistances += YaverDis;//Y方向距离总值
		}
		//角度均值
		double Ydismean = Ydistances / (linenum - 1);//Y方向距离均值
		double radsum = std::accumulate(rads.begin(), rads.end(), 0.0);
		double radmean = radsum / rads.size(); //角度均值均值
		double Vy = Vyvars / (linenum - 1);
		//std::cout << "Vy：" << Vy << std::endl;
		double Vx = Vy * std::sin(radmean);
		//double Vx = 0.000093;
		//double Vx = Vy * 85.9503 / Ydismean;
		//std::cout << "Vx：" << Vx << std::endl;
		double High = abs(Ydismean * std::sin(45 * CV_PI / 180));//垫高的值求解Ks
		vector<double>Highs;
		for (int i = 1; i < linenum; i++)
		{
			double high = High * i;
			Highs.push_back(high);
		}
		//求解每张图的K值
		std::vector <cv::Point3d> Ks;
		std::vector<double>Distance;//标定板移动距离
 		SolveK(Static_World_Points, pix_points, Ks);
		Eigen::Matrix2f Cam_internal;
		std::vector< Eigen::Matrix2f>Cam_Poses;
		SolveLine_Calibrationl(disssss, Ks, Cam_internal, Cam_Poses, High);
		//double Vx = Vy * (Static_World_Points[1][35] - Static_World_Points[0][35]) / Ydismean;	//求解Vz
		double Zsum = 0;
		for (int i = 0; i < linenum - 2; i++)
		{
			double val = Cam_Poses[i](1, 1) - Cam_Poses[i + 1](1, 1);
			Zsum += val;
		}
		double Zdismean = Zsum / (linenum - 1);
		double Vz = Vy * Zdismean / Ydismean;
		//double Vz = Vy * Cam_Poses[0](1, 1) / Ydismean;
		std::cout << "Vx：" << Vx << std::endl;
		std::cout << "Vy：" << Vy << std::endl;
		std::cout << "Vz：" << Vz << std::endl;
		for (int i = 0; i < pix_points.size(); i++)
		{
			for (int j = 1; j < pix_points[i].size(); j++)
			{
				pix_points[i].erase(pix_points[i].begin() + j);
				j + 1;
			}
			pix_points[i].erase(pix_points[i].end() - 2, pix_points[i].end());
		}


		//double params[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
		////double params[8] = { 0,0,0,0,0,0,0,0 };
		//ceres::Problem problem;
		//for (int i = 0; i < World_Points.size(); i++)
		//{
		//	for (int j = 0; j < World_Points[0].size(); j++)
		//    {
		//		ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<MFunctor, 2, 11>(new MFunctor(pix_points[i][j].x, pix_points[i][j].y, New_Points[i][j].x, New_Points[i][j].y, New_Points[i][j].z));
		//		//ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<MFunctor, 2, 8>(new MFunctor(pix_points[i][j].x, pix_points[i][j].y, World_Points[i][j].x, World_Points[i][j].y, World_Points[i][j].z));
		//		//三个参数分别为代价函数、核函数和待估参数
		//		problem.AddResidualBlock(cost_function, NULL, params);
		//    }
		//}
		//// 第二步，配置Solver
		//ceres::Solver::Options options;
		////配置增量方程的解法
		//options.max_num_iterations = 1000;
		//options.linear_solver_type = ceres::DENSE_QR;
		////是否输出到cout
		//options.minimizer_progress_to_stdout = true;
		////第三步，创建Summary对象用于输出迭代结果
		//ceres::Solver::Summary summary;
		////第四步，执行求解
		//ceres::Solve(options, &problem, &summary);
		////第五步，输出求解结果
		//std::cout << summary.BriefReport() << endl;
		////std::cout << "m11:" << params[0] << endl;
		////std::cout << "m12:" << params[1] << endl;
		////std::cout << "m13:" << params[2] << endl;
		////std::cout << "m14:" << params[3] << endl;
		////std::cout << "m21:" << params[4] << endl;
		////std::cout << "m22:" << params[5] << endl;
		////std::cout << "m23:" << params[6] << endl;
		////std::cout << "m24:" << params[7] << endl;
		////std::cout << "m31:" << params[8] << endl;
		////std::cout << "m32:" << params[9] << endl;
		////std::cout << "m33:" << params[10] << endl;
		//
		//Eigen::Matrix<double, 3, 4> M;
		//M << params[0], params[1], params[2], params[3],
		//	params[4], params[5], params[6], params[7],
		//	params[8], params[9], params[10], 1;

		//std::cout << "M:\n" << M << endl << endl;
		//Eigen::Matrix<double, 4, 1> XYZ;
		//XYZ << World_Points[1][2].x,
		//	0,
		//	World_Points[1][2].z,
		//	1;

		//Matrix<double, 3, 1> result1 = M * XYZ;
		//cout << "Pointxyz:\n" << result1 << endl << endl;

		Eigen::Matrix<double, 3, 3> M1;
		M1 << Cam_internal(0, 0), 0.0, Cam_internal(0, 1),
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0;
		std::cout << "M1:\n" << M1 << endl << endl;
		Eigen::Matrix<double, 3, 3> M2;
		M2 << 1.0, -Vx / Vy, 0.0,
			0.0, 1.0 / Vy, 0.0,
			0.0, Vz / Vy, 1.0;
		//std::cout << "M2:\n" << M2 << endl << endl;
		//Eigen::Matrix<double, 3, 3> M3= M1* M2;
		//std::cout << "M3:\n" << M3 << endl;
		//Eigen::Matrix<double, 3, 4> Pose = M2.inverse()*M1.inverse()*M ;
		////cout <<"(M2* M1).invers:\n" << M2.inverse() * M1.inverse() << endl;
		//Eigen::Matrix<double, 3, 3> R= Pose.leftCols(3);
		//Eigen::Matrix<double, 3, 1> RT= Pose.rightCols(1);
		//Eigen::Matrix<double, 3, 1> T = R.inverse() * RT;
		//cout << "Pose:\n" << Pose << endl << endl;
		//cout << "T:\n" << T << endl << endl;

		//Eigen::Matrix<double, 3, 3> M = M1 * M2;
		//std::cout << "M:\n" << M<< endl;
		//double params[5] = { 0,0,0,0,0};
		////double params[8] = { 0,0,0,0,0,0,0,0 };
		//ceres::Problem problem;
		//for (int i = 0; i < World_Points.size(); i++)
		//{
		//	for (int j = 0; j < New_Points[0].size(); j++)
		//    {

		//		ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<MFunctor, 4, 5>(new MFunctor(pix_points[i][j], New_Points[i][j], M, Cam_Poses[0]));
		//		//三个参数分别为代价函数、核函数和待估参数
		//		problem.AddResidualBlock(cost_function, NULL, params);
		//    }
		//}
		//// 第二步，配置Solver
		//ceres::Solver::Options options;
		////配置增量方程的解法
		//options.max_num_iterations = 1000;
		////options.linear_solver_type = ceres::DENSE_QR;
		//options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		////options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		////是否输出到cout
		//options.minimizer_progress_to_stdout = true;
		////第三步，创建Summary对象用于输出迭代结果
		//ceres::Solver::Summary summary;
		////第四步，执行求解
		//ceres::Solve(options, &problem, &summary);
		////第五步，输出求解结果
		//std::cout << summary.BriefReport() << endl;
		//Eigen::Matrix<double, 3, 4>Pose;
		//Pose << 
		//	(ceres::cos(params[0]) * ceres::cos(params[1])),
		//	(ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) - ceres::sin(params[0]) * ceres::cos(params[2])),
		//	(ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) + ceres::cos(params[0]) * ceres::sin(params[2])),
		//	((ceres::cos(params[0]) * ceres::cos(params[1])) * Cam_Poses[0](0, 1)
		//		+ (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) - ceres::sin(params[0]) * ceres::cos(params[2])) * params[3] 
		//		+ (ceres::cos(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) + ceres::cos(params[0]) * ceres::sin(params[2])) * Cam_Poses[0](1, 1)),
		//	
		//	(ceres::sin(params[0]) * ceres::cos(params[1])),
		//	(ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) + ceres::cos(params[0]) * ceres::cos(params[2])),
		//	(ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])),
		//	((ceres::sin(params[0]) * ceres::cos(params[1])) * Cam_Poses[0](0, 1) + (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::sin(params[2]) 
		//		+ ceres::cos(params[0]) * ceres::cos(params[2])) * params[3] 
		//		+ (ceres::sin(params[0]) * ceres::sin(params[1]) * ceres::cos(params[2]) - ceres::cos(params[0]) * ceres::sin(params[2])) * Cam_Poses[0](1, 1)),
		//    
		//	(-ceres::sin(params[1])),
		//	(ceres::cos(params[1]) * ceres::sin(params[2])),
		//	(ceres::cos(params[1]) * ceres::cos(params[2])),
		//	((-ceres::sin(params[1])) * Cam_Poses[0](0, 1) + (ceres::cos(params[1]) * ceres::sin(params[2])) * params[3] + (ceres::cos(params[1]) * ceres::cos(params[2])) * Cam_Poses[0](1, 1));
		//cout << "Pose:\n" << Pose << endl;
		//cout << "W:\n" << params[4] << endl;
		//Eigen::Matrix<double, 3, 3> R = Pose.leftCols(3);
		//Eigen::Matrix<double, 3, 1> T = R.inverse() * Pose.rightCols(1);
		//cout << "T:\n" << T << endl << endl;
	//}
	cv::waitKey(0);
	return 0;
}