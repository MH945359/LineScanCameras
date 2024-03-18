#include"Static_Calibration.h"
using namespace std;
using namespace Eigen;

struct MFunctor
{

	double U_guan, V_guan, Xw_guan, Yw_guan, Zw_guan;

	//构造函数，用已知的x、y数据对其赋值
	 //CostFunctor(double xc, double u)
	MFunctor(double u, double v, double xw, double yw, double zw)
	{
		U_guan = u;
		V_guan = v;
		Xw_guan = xw;
		Yw_guan = yw;
		Zw_guan = zw;
	}
	//重载括号运算符，两个参数分别是估计的参数和由该参数计算得到的残差
	template <typename T>
	bool operator()(const T* const params, T* residual)const
	{
		residual[0] = ((params[0] * T(Xw_guan) + params[1] * T(Yw_guan) + params[2] * T(Zw_guan) + params[3]) / (params[8] * T(Xw_guan) + params[9] * T(Yw_guan) + params[10] * T(Zw_guan) + 1.0)) - T(U_guan);
		residual[1] = (params[4] * T(Xw_guan) + params[5] * T(Yw_guan) + params[6] * T(Zw_guan) + params[7]) - T(V_guan);
		/*residual[0] =((M(0,0)* params[0]+ M(0, 1) * params[1]+ M(0, 2) * params[2]) * T(Xw_guan)
			        + (M(0, 0) * params[3] + M(0, 1) * params[4] + M(0, 2) * params[5]) * T(Yw_guan) 
			        + (M(0, 0) * params[6] + M(0, 1) * params[7] + M(0, 2) * params[8]) * T(Zw_guan)
			        + (M(0, 0) * params[9] + M(0, 1) * params[10] + M(0, 2) * 1.0))/
			         ((M(2, 1) * params[1] + params[2])* T(Xw_guan)+ (M(2, 1) * params[4] + params[5]) * T(Yw_guan)+ (M(2, 1) * params[7] + params[8]) * T(Zw_guan)+ (M(2, 1) * params[10] + 1.0))- T(U_guan);

		residual[1] = (params[1] * T(Xw_guan) + params[4] * T(Yw_guan) + params[7] * T(Zw_guan) + params[10])-T(V_guan);
		residual[2] = params[0] * params[1] + params[4] * params[5] + params[8] * params[9];*/
		return true;
	}
};

int main()
{
	std::ofstream outputFile(R"(D:\桌面\point3d_data.txt)");
	std::ofstream putFile(R"(D:\桌面\point3d_data1.txt)");
	//cv::Mat img = cv::imread(R"(F:\Line_Picture\24.1.8Line\3.bmp)", cv::IMREAD_GRAYSCALE);
	//cv::Mat img = cv::imread(R"(D:\test_pictures\Line_Picture\24.1.24\2.bmp)", cv::IMREAD_GRAYSCALE);
	//cv::Mat img = cv::imread(R"(D:\test_pictures\Line_Picture\24.1.15_linepicture\2.bmp)", cv::IMREAD_GRAYSCALE);
	//cv::Mat img = cv::imread(R"(F:\Line_Picture\24.1.22\11.bmp)", cv::IMREAD_GRAYSCALE);
	cv::Mat img = cv::imread(R"(D:\test_pictures\Line_Picture\24.2.26\4.bmp)", cv::IMREAD_GRAYSCALE);
	int coline =800;
	bool sss = false;
	//获取像素坐标系角点坐标（x,y）
	//选择行数复制500行生成图片
	int linenum = 6;
	//每个矩形长度
	double Wide = 8.0;
	double Long = 40.0;
	
	//int linenum = 0;
	//cout << "Select Line Number: ";//提取几条线
	//cin >> linenum;
	//int coline = 0;
	//cout << "Select Row: ";//提取第多少行的像素
	//cin >> coline;
	int jiange =3/ 0.026;
	for (int jjj = 0; jjj < 100; jjj++)
	{
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
			cv::Mat Angle = (cv::Mat_<double>(3, 1) <<45 * CV_PI / 180, 0, -rads[i]);
			cv::Rodrigues(Angle, Rmatrix);
			for (int j = 0; j < Progress_Points[0].size(); j++)
			{
				cv::Mat original = (cv::Mat_<double>(3, 1) << Progress_Points[i][j].x, Progress_Points[i][j].y, Progress_Points[i][j].z);
				cv::Mat Newpoint_MAT= Rmatrix*original;
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
				val1.x = Progress_Points[i + 1][j].x / std::cos(rads[i]) +(Progress_Points[i + 1][j].y - Progress_Points[0][j].y) * std::tan(rads[i]);
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
					double Dxval = New_Points[i][j].x - (New_Points[i + 1][j].x + abs(New_Points[i][j].y - New_Points[i + 1][j].y)  * factor);
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
			})/ DxyzMean.size();
		for (int i = 0; i < linenum-1; i++)
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
			double U0 = Cam_internals[i](0, 1) - 4096;
			if (-30 < U0 && U0 <0 && Cam_internals[i](0, 0) < 12000)
			{
				sss = true;
				std::cout << "Cam_internal:" << std::endl;
				cout << Cam_internals[i] << endl;
				cout << endl;
				std::cout << "Cam_Pose No." << i+1 << ":" << std::endl;
				std::cout << Cam_Poses[2 * i] << std::endl;
				cout << endl;
				std::cout << "Cam_Pose No." << i +2 << ":" << std::endl;
				std::cout << Cam_Poses[2 * i + 1] << std::endl;
				cout << endl;
				cout << "Vx:" << Vx << endl;
				cout << "Vy:" << Vy << endl;
				cout << "Vz:" << Vz << endl;
				aa += i;
				ma = aa;
			}
		}
		vector<double>pix_vec;
		double pixSum = 0;
		for (int i = 0; i < Static_World_Points[0].size()/2; i++)
		{
			pixSum+=pix_points[ma][2 * i + 2].x - pix_points[ma][2 * i].x;

		}
		double  pixMean = 2 * pixSum / Static_World_Points[0].size();
		if (sss == false)
		{
			std::cout << "No Optimal Solution" << std::endl;
			coline -=jiange * (linenum) - 3;
			goto MH1;
		}
		else
		{
			cout <<"起始位置： "<< coline - linenum * jiange << endl;
			cout <<"NO. " << ma << endl;
			cout << endl;
			for (int i = 0; i < Static_World_Points[0].size(); i++)
			{
				if(i%2==0)
				{ 
				Eigen::Vector2d vec;
				vec << Static_World_Points[ma][i], 1;
				Eigen::Vector2d pix_result = Cam_internals[ma] * Cam_Poses[2 * (ma)] * vec;
				pix_vec.push_back(pix_result(0, 0) / pix_result(1, 0));
				}
			}
			double error = 0;
			for (int i = 0; i < pix_vec.size()-1; i++)
			{
				error += (pix_vec[i + 1] - pix_vec[i] - pixMean ) / pixMean;
			}
			cout << "误差： " << error / pix_vec.size() << endl;
		   
			goto MH;
			exit(0);
		}
	MH: {
		Eigen::Matrix3d M;
		Eigen::Matrix3d M1;
		Eigen::Matrix3d M2;
		M1 << Cam_internals[ma](0, 0), 0, Cam_internals[ma](0, 1),
			0, 1, 0,
			0, 0, 1;
		M2 << 1, -Vx / Vy, 0,
			0, 1 / Vy, 0,
			0, -Vz / Vy, 1;
		M = M1 * M2;
		cv::Mat M_cv = (cv::Mat_<double>(3, 3, CV_64F) <<
			M(0, 0), M(0, 1), M(0, 2),
			M(1, 0), M(1, 1), M(1, 2),
			M(2, 0), M(2, 1), M(2, 2));
		for (int i = 0; i < World_Points.size(); i++)
		{
			for (int j = 0; j < 4; j++)
			{
				pix_points[i].erase(pix_points[i].end() - 1, pix_points[i].end());
			}
		}
		for (int i = 0; i < World_Points.size(); i++)
		{
			for (int j = 0; j < World_Points[0].size(); j++)
			{
				outputFile << World_Points[i][j].x << " " << World_Points[i][j].y << " " << World_Points[i][j].z << std::endl;
				putFile << pix_points[i][j].x << " " << pix_points[i][j].y << std::endl;

			}

		}
		outputFile.close(); 
			vector< vector<cv::Point3d>>sss_points;
		for (int i = 0; i < New_Points.size(); i++)
		{
			vector<cv::Point3d>sss_point;
			for (int j=0;j< New_Points[0].size();j++)
			{
				cv::Point3d points;
				points.x = New_Points[i][j].x - New_Points[i][0].x;
				points.y = New_Points[i][j].y;
				points.z = New_Points[i][j].z;
				sss_point.push_back(points);
			}
			sss_points.push_back(sss_point);
		}


		double paramss[11] = { 0,0,0,4000,0,0,0,0,0,0,0 };
		ceres::Problem problem;
		/*for (int i = ma; i < ma+1; i++)
		{*/
		for (int i = 0; i < World_Points.size(); i++)
		{

			for (int j = 0; j < World_Points[0].size(); j++)
			{
				if (j % 2 == 0)
				{
					ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<MFunctor, 2, 11>(new MFunctor(pix_points[i][j].x, pix_points[i][j].y, New_Points[i][j].x, New_Points[i][j].y, New_Points[i][j].z));
					//ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<MFunctor, 2, 8>(new MFunctor(pix_points[i][j].x, pix_points[i][j].y, World_Points[i][j].x, World_Points[i][j].y, World_Points[i][j].z));
					//三个参数分别为代价函数、核函数和待估参数
					problem.AddResidualBlock(cost_function, NULL, paramss);
				}
			}

		}
		// 第二步，配置Solver
		ceres::Solver::Options options;
		//配置增量方程的解法
		options.max_num_iterations = 1000;
		options.linear_solver_type = ceres::DENSE_QR;
		//是否输出到cout
		options.minimizer_progress_to_stdout = true;
		//第三步，创建Summary对象用于输出迭代结果
		ceres::Solver::Summary summary;
		//第四步，执行求解
		ceres::Solve(options, &problem, &summary);
		//第五步，输出求解结果
		std::cout << summary.BriefReport() << endl;
		//std::cout << "m11:" << params[0] << endl;
		//std::cout << "m13:" << params[2] << endl;
		//std::cout << "m14:" << params[3] << endl;
		//std::cout << "m21:" << params[4] << endl;
		//std::cout << "m22:" << params[5] << endl;
		//std::cout << "m23:" << params[6] << endl;
		//std::cout << "m24:" << params[7] << endl;
		//std::cout << "m31:" << params[8] << endl;
		//std::cout << "m32:" << params[9] << endl;
		//std::cout << "m33:" << params[10] << endl;

		Eigen::Matrix<double, 3, 4> MM;
		MM << paramss[0], paramss[1], paramss[2], paramss[3],
			paramss[4], paramss[5], paramss[6], paramss[7],
			paramss[8], paramss[9], paramss[10], 1.0;

		std::cout << "MM:\n" << MM << endl << endl;
		for (int j = 0; j < World_Points[0].size(); j++)
		{
			if (j % 2 == 0)
			{
				Eigen::Matrix<double, 4, 1> XYZ;
				XYZ << New_Points[3][j].x,
					New_Points[3][j].y,
					New_Points[3][j].z,
					1.0;

				Matrix<double, 3, 1> result1 = MM * XYZ;
				double pix_valx=result1(0, 0) - pix_points[3][j].x;
				double pix_valy= result1(1, 0) - pix_points[3][j].y;
				cout << "Pointxyz:\n" << result1 << endl << endl;
				cout << "val:\n" << pix_valx << endl << pix_valy << endl << endl;
			}
		}
		exit(0);
		}
     MH1:
	{}
	}


	std::system("cls");
	cv::waitKey(0);
	return 0;
}