#include"Static_Calibration.h"
using namespace std;
using namespace Eigen;

int main()
{
	cv::Mat img = cv::imread(R"(D:\test_pictures\Line_Picture\24.2.26\4.bmp)", cv::IMREAD_GRAYSCALE);
	int coline = 800;

	//获取像素坐标系角点坐标（x,y）
	//选择行数复制500行生成图片
	int linenum = 6;
	//每个矩形长度
	double Wide = 8.0;
	double Long = 40.0;
	int jiange = 3 / 0.026;
	cout << "coline: " << coline << endl;
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
	vector <vector < cv::Point3d >> Progress_Points;//世界坐标系角点三维坐标
	vector < cv::Point3d > Progress_Point;
	vector<double>Intercepts;//拟合线截距
	vector<float>rads;//斜率角度
	for (int i = 0; i < linenum; i++)
	{
		cv::Vec4f lineParams;
		cv::fitLine(HypPoints[i], lineParams, cv::DIST_L2, 0, 0.001, 0.001);//通过最小二乘法拟合斜边点的方程
		float k = lineParams[1] / lineParams[0];
		double b = lineParams[3] - k * lineParams[2];
		Intercepts.push_back(b);
		float rad = std::atan(k);
		rads.push_back(rad);
		int a = 0;
		for (int j = 0; j < VerPoint.size() - 4; j++)
		{
			if (j % 2 == 0)
			{
				//竖边世界坐标点
				cv::Point3d val;
				val.y = Intercepts[i] - Intercepts[0];
				//val.y = k * VerPoint[j] + b - Intercepts[0];
				val.x = VerPoint[j];
				//val.z = 2 * val.y * std::cos(45 * CV_PI / 180) * std::cos(45 * CV_PI / 180);
				val.z = 0;
				Progress_Point.push_back(val);
				a++;
			}
			else
			{
				cv::Point3d val1;
				val1.y = Intercepts[i] - Intercepts[0];
				//val1.y = HypPoints[i][a - 1].y - Intercepts[0];
				val1.x = HypPoints[i][a - 1].x;
				//val1.z = 2 * val1.y * std::cos(45 * CV_PI / 180) * std::cos(45 * CV_PI / 180);
				val1.z = 0;
				Progress_Point.push_back(val1);
			}
		}
		Progress_Points.push_back(Progress_Point);
		Progress_Point.clear();

	}
	cv::Mat Rmatrix;
	vector<vector<cv::Point3d>>New_Points;
	for (int i = 0; i < Progress_Points.size(); i++)
	{
		vector<cv::Point3d>New_Point;
		cv::Mat Angle = (cv::Mat_<double>(3, 1) << 45 * CV_PI / 180, 0, -rads[i]);
		cv::Rodrigues(Angle, Rmatrix);
		for (int j = 0; j < Progress_Points[0].size(); j++)
		{
			cv::Mat original = (cv::Mat_<double>(3, 1) << Progress_Points[i][j].x, Progress_Points[i][j].y, Progress_Points[i][j].z);
			cv::Mat Newpoint_MAT = Rmatrix * original;
			cv::Point3d changePoint;
			changePoint.x = Newpoint_MAT.at<double>(0, 0);
			changePoint.y = Newpoint_MAT.at<double>(1, 0);
			changePoint.z = Newpoint_MAT.at<double>(2, 0);
			New_Point.push_back(changePoint);
		}
		New_Points.push_back(New_Point);
	}
	vector <vector < cv::Point3d >> World_Points;//以第一条像素为世界坐标系基准建立世界坐标系
	vector < cv::Point3d >World_Point;
	for (int i = 0; i < Progress_Points[0].size(); i++)//建立第一条线的坐标系
	{
		cv::Point3d val;
		val.y = 0;
		val.x = Progress_Points[0][i].x / std::cos(rads[0]);
		val.z = 0;
		World_Point.push_back(val);
	}
	World_Points.push_back(World_Point);
	World_Point.clear();
	for (int i = 0; i < Progress_Points.size() - 1; ++i)
	{
		for (int j = 0; j < Progress_Points[0].size(); j++)
		{

			cv::Point3d val1;
			val1.y = (Progress_Points[i + 1][j].y - Progress_Points[0][j].y) / std::cos(rads[i]) * std::cos(45 * CV_PI / 180);
			//val1.x = Progress_Points[i + 1][j].x+ val1.y *std::tan (rads[i]);
			val1.x = Progress_Points[i + 1][j].x / std::cos(rads[i]) + (Progress_Points[i + 1][j].y - Progress_Points[0][j].y) * std::tan(rads[i]);
			val1.z = val1.y * std::tan(45 * CV_PI / 180);
			World_Point.push_back(val1);


		}
		World_Points.push_back(World_Point);
		World_Point.clear();
	}

	//求解Vy;
	vector<cv::Point3d>DxyzMean;
	double Vysum = 0;
	double Vxsum = 0;
	double Vzsum = 0;
	double factor = std::pow(Wide * Wide / (Wide * Wide + Long * Long), 0.5);
	for (int i = 0; i < linenum - 1; i++)
	{
		double Dxsum = 0;
		double Dysum = 0;
		double DZsum = 0;
		for (int j = 0; j < World_Points[0].size(); j++)
		{
			if (j % 2 == 1)
			{
				//double Dxval = World_Points[i][j].x - (World_Points[i + 1][j].x + abs(World_Points[i][j].z - World_Points[i + 1][j].z) / std::cos(45 * CV_PI / 180)/8);
				double Dxval = New_Points[i][j].x - (New_Points[i + 1][j].x + abs(New_Points[i][j].y - New_Points[i + 1][j].y) * factor);
				//double Dxval =abs(New_Points[i][j].x - New_Points[i + 1][j].x);
				Dxsum += Dxval;
			}
			double Dyval = New_Points[i + 1][j].y - New_Points[i][j].y;
			double DZval = New_Points[i + 1][j].z - New_Points[i][j].z;
			DZsum += DZval;
			Dysum += Dyval;
		}
		cv::Point3d Mean;
		Mean.x = Dxsum / New_Points[0].size();//DxMean
		Mean.y = Dysum / New_Points[0].size();//DyMean
		Mean.z = DZsum / New_Points[0].size();//DzMean
		double Vyvar = Mean.y / jiange;
		Vysum += Vyvar;
		DxyzMean.push_back(Mean);
	}
	vector<double>DzMean;
	double sumZ = std::accumulate(DxyzMean.begin(), DxyzMean.end(), 0.0,
		[](double partialSum, const cv::Point3d& point)
		{
			return partialSum + point.z;
		}) / DxyzMean.size();
		for (int i = 0; i < linenum - 1; i++)
		{
			double zsum = 0;
			for (int j = 0; j < New_Points[0].size(); j++)
			{
				double zval = New_Points[i + 1][j].z - (i + 1) * sumZ;
				zsum += zval;
			}
			DzMean.push_back(zsum / New_Points[0].size());
		}
		double Vy = Vysum / DxyzMean.size();
		for (int i = 0; i < DxyzMean.size(); i++)
		{
			double Vz = Vy * DzMean[i] / DxyzMean[i].y;
			double Vx = Vy * DxyzMean[i].x / DxyzMean[i].y;
			Vxsum += Vx;
			Vzsum += Vz;
		}
		double Vx = Vxsum / DxyzMean.size();
		double Vz = Vzsum / DxyzMean.size();
		//静态标定
		vector<vector<double>>Static_World_Points; //静态标定的世界坐标点
		CalibrationConver(linenum, VerPoint, HypPoints, Static_World_Points);//动态标定板坐标转换到静态标定的世界坐标点
		std::vector <cv::Point3d> Ks;
		SolveK(Static_World_Points, pix_points, Ks);
		std::vector< Eigen::Matrix2d>Cam_internals;
		std::vector< Eigen::Matrix2d>Cam_Poses;
		SolveLine_Calibration(DxyzMean, Ks, Cam_internals, Cam_Poses);
		//SolveLine_Calibration(DxyzMean, Ks, Cam_internals, Cam_Poses,3.15, 0.133800094910239);
		std::system("cls");
		int ma = 0;
		for (int i = 0; i < Cam_internals.size(); i++)
		{
				int aa = 0;
				std::cout << "Cam_internal:" << std::endl;
				cout << Cam_internals[i] << endl;
				cout << endl;
				std::cout << "Cam_Pose No." << i + 1 << ":" << std::endl;
				std::cout << Cam_Poses[2 * i] << std::endl;
				cout << endl;
				std::cout << "Cam_Pose No." << i + 2 << ":" << std::endl;
				std::cout << Cam_Poses[2 * i + 1] << std::endl;
				cout << endl;
				cout << "Vx:" << Vx << endl;
				cout << "Vy:" << Vy << endl;
				cout << "Vz:" << Vz << endl;
				aa += i;
				ma = aa;
				
		}
		vector<double>pix_vec;
		double pixSum = 0;
		for (int i = 0; i < Static_World_Points[0].size() / 2; i++)
		{
			pixSum += pix_points[ma][2 * i + 2].x - pix_points[ma][2 * i].x;

		}
		double  pixMean = 2 * pixSum / Static_World_Points[0].size();
			
		for (int i = 0; i < Static_World_Points[0].size(); i++)
		{
			if (i % 2 == 0)
			{
				Eigen::Vector2d vec;
				vec << Static_World_Points[ma][i], 1;
				Eigen::Vector2d pix_result = Cam_internals[ma] * Cam_Poses[2 * (ma)] * vec;
				pix_vec.push_back(pix_result(0, 0) / pix_result(1, 0));
			}
		}
		double error = 0;
		for (int i = 0; i < pix_vec.size() - 1; i++)
		{
			error += (pix_vec[i + 1] - pix_vec[i] - pixMean) / pixMean;
		}
		cout << "误差： " << error / pix_vec.size() << endl;


	std::system("cls");
	cv::waitKey(0);
	return 0;
}