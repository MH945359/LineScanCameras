#define _CRT_SECURE_NO_WARNINGS 
#include"Static_Calibration.h"

void AreaSelect(int& number, cv::Mat& stats, cv::Mat& centroids, std::vector<cv::Point2d>& pix_point, int& colines)
{
	std::vector<int>Area;
	for (int i = 0; i < number; i++)
	{
		if (stats.at<int>(i, 4) < 1000)
		{
			Area.push_back(0);
		}
		else
		{
			Area.push_back(stats.at<int>(i, 4));
		}
	}
	int j = 0;
	for (int i = 0; i < number; i++)
	{
		int val = 0;
		if (Area[i] != 0)
		{

			int power = static_cast<int>(std::pow(10, std::floor(std::log10(Area[i]))));
			val = round(Area[i] / power) * power;
		}
		if (val / 1000 > 20)
		{
			Area[i] = 0;
		}
		if (Area[i] != 0)
		{
			j++;
		}
	}
	int Areasum = accumulate(Area.begin(), Area.end(), 0.0);
	float Areamean = Areasum / j;
	for (int i = 0; i < number; i++)
	{
		if (Area[i] != 0)
		{
			float sub = abs(Area[i] - Areamean) / Areamean;
			//std::cout << "No." << i << ": " << sub << endl;
			if (sub > 0.3)
			{
				Area[i] = 0;
			}
		}
	}
	for (int i = 0; i < number; i++)
	{
		if (Area[i] != 0)
		{
			float sub = abs(stats.at<int>(i, 4) - Areamean) / Areamean;
			if (sub < 0.3)
			{
				cv::Point2d point;
				point.x = centroids.at<double>(i, 0);
				point.y = colines;
				pix_point.push_back(point);
			}
		}
	}
}
void CalibrationConver(int linenum, std::vector<double>& VerPoint, std::vector<std::vector<cv::Point2d>>& HypPoints, std::vector<std::vector<double>>& Static_World_Points)
{
	for (int i = 0; i < linenum; i++)
	{
		std::vector <double> World_Point;
		cv::Vec4f lineParams;
		cv::fitLine(HypPoints[i], lineParams, cv::DIST_L2, 0, 0.01, 0.01);//通过最小二乘法拟合斜边点的方程
		float k = lineParams[1] / lineParams[0];
		float rad = std::atan(k);
		int a = 0;
		for (int j = 0; j < VerPoint.size() - 4; j++)
		{
			if (j % 2 == 0)
			{
				//竖边世界坐标点
				double val=  VerPoint[j] / cos(rad);
				World_Point.push_back(val);
				a++;
			}
			else
			{
				double vall = HypPoints[i][a-1].x / cos(rad);
				World_Point.push_back(vall);
			}
		}
		Static_World_Points.push_back(World_Point);
		World_Point.clear();
	}
}
void SolveK(std::vector<std::vector<double>>& Static_World_Points,std::vector<std::vector<cv::Point2d>>&pix_points, std::vector <cv::Point3d>&Ks)
{
	for (int j = 0; j < Static_World_Points.size(); j++)
	{
		double params[3] = { 0,0,0 };
		ceres::Problem problem;

		for (int i = 0; i < Static_World_Points[0].size(); i++)
		{
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<KFunctor, 1, 3>(new KFunctor( Static_World_Points[j][i], pix_points[j][i].x));
			//三个参数分别为代价函数、核函数和待估参数
			problem.AddResidualBlock(cost_function, NULL, params);
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
		std::cout << summary.BriefReport() << std::endl; // 添加此处

		std::cout << "K1:" << params[0] << std::endl;
		std::cout << "K2:" << params[1] << std::endl;
		std::cout << "K3:" << params[2] << std::endl;
		cv::Point3d k;
		k.x = params[0];//K1
		k.y = params[1];//K2
		k.z = params[2];//K3
		Ks.push_back(k);
	}
}
void SolveLine_Calibration(std::vector<cv::Point3d>& Dxyz, std::vector <cv::Point3d>& Ks, std::vector<Eigen::Matrix2d>& Cam_internals, std::vector<Eigen::Matrix2d>& Cam_Poses)
//void SolveLine_Calibration(std::vector<cv::Point3d>&Dxyz,std::vector <cv::Point3d>& Ks, std::vector<Eigen::Matrix2d>&Cam_internals, std::vector<Eigen::Matrix2d>&Cam_Poses,double high, double h)
{
	for (int k = 0; k < Dxyz.size(); k++)
	{
		double paramsl[4] = { 10000,1,4096,0 };
		ceres::Problem probleml;
		for (int i = k; i < k+1; i++)
		{
			ceres::CostFunction* cost_functionl = new ceres::AutoDiffCostFunction<LineFunctor, 4, 4>(new LineFunctor(Ks[i].x, Ks[i].y, Ks[i].z, Ks[i + 1].x, Ks[i + 1].y, Ks[i + 1].z, Dxyz[i].z, Dxyz[i].x));
			//ceres::CostFunction* cost_functionl = new ceres::AutoDiffCostFunction<LineFunctor, 4, 4>(new LineFunctor(Ks[i].x, Ks[i].y, Ks[i].z, Ks[i + 1].x, Ks[i + 1].y, Ks[i + 1].z, high, h));
			//三个参数分别为代价函数、核函数和待估参数
			probleml.AddResidualBlock(cost_functionl, NULL, paramsl);
		}
		// 第二步，配置Solver  
		ceres::Solver::Options optionsl;
		//配置增量方程的解法
		optionsl.max_num_iterations = 10000;
		optionsl.linear_solver_type = ceres::DENSE_QR;
		//是否输出到cout
		optionsl.minimizer_progress_to_stdout = true;
		//第三步，创建Summary对象用于输出迭代结果
		ceres::Solver::Summary summaryl;
		//第四步，执行求解
		ceres::Solve(optionsl, &probleml, &summaryl);
		std::cout << summaryl.BriefReport() << std::endl; // 添加此处
		//内参
		Eigen::Matrix2d Cam_internal;
		std::cout << k << std::endl;
		Cam_internal(0, 0) = paramsl[0];//Fu
		Cam_internal(0, 1) = paramsl[2];//U0
		Cam_internal(1, 0) = 0;
		Cam_internal(1, 1) = 1;
		Cam_internals.push_back(Cam_internal);
		std::cout << std::endl;
		std::cout <<"No." <<k+1<< "Cam_internal:" << std::endl;
		std::cout << Cam_internal << std::endl;
		std::cout << std::endl;
		//外参
		for (int i = k; i <k+2; i++)
		{
			Eigen::Matrix2d Cam_Pose;
			double Tz = paramsl[0] * paramsl[1] / (paramsl[2] * Ks[i].y + Ks[i].x);
			double Tx = (Ks[i].z - paramsl[2]) * paramsl[1] / (paramsl[0] * Ks[i].y + Ks[i].x);
			Cam_Pose(0, 0) = paramsl[1];//r11
			Cam_Pose(0, 1) = Tx;
			Cam_Pose(1, 0) = paramsl[3];//r31
			Cam_Pose(1, 1) = Tz;
			std::cout << "Cam_Pose No." << i + 1 << ":" << std::endl;
			std::cout << Cam_Pose << std::endl;
			std::cout << std::endl;
			Cam_Poses.push_back(Cam_Pose);
		}

	}


}
void SolveLine_Calibrationl(std::vector<double>& Dxyz, std::vector <cv::Point3d>& Ks, Eigen::Matrix2f& Cam_internal, std::vector<Eigen::Matrix2f>& Cam_Poses,double &high)
{
	for (int k = 0; k < Dxyz.size(); k++)
	{
		double paramsl[4] = { 10000,1,4096,0 };
		ceres::Problem probleml;
		for (int i = k; i < k + 1; i++)
		{
			ceres::CostFunction* cost_functionl = new ceres::AutoDiffCostFunction<LineFunctor, 4, 4>(new LineFunctor(Ks[i].x, Ks[i].y, Ks[i].z, Ks[i + 1].x, Ks[i + 1].y, Ks[i + 1].z, high, Dxyz[i]));

			//三个参数分别为代价函数、核函数和待估参数
			probleml.AddResidualBlock(cost_functionl, NULL, paramsl);
		}
		// 第二步，配置Solver  
		ceres::Solver::Options optionsl;
		//配置增量方程的解法
		optionsl.max_num_iterations = 10000;
		optionsl.linear_solver_type = ceres::DENSE_QR;
		//是否输出到cout
		optionsl.minimizer_progress_to_stdout = true;
		//第三步，创建Summary对象用于输出迭代结果
		ceres::Solver::Summary summaryl;
		//第四步，执行求解
		ceres::Solve(optionsl, &probleml, &summaryl);
		std::cout << summaryl.BriefReport() << std::endl; // 添加此处
		//内参
		std::cout << k << std::endl;
		Cam_internal(0, 0) = paramsl[0];//Fu
		Cam_internal(0, 1) = paramsl[2];//U0
		Cam_internal(1, 0) = 0;
		Cam_internal(1, 1) = 1;
		std::cout << std::endl;
		std::cout << "No." << k + 1 << "Cam_internal:" << std::endl;
		std::cout << Cam_internal << std::endl;
		std::cout << std::endl;
		//外参
			Eigen::Matrix2f Cam_Pose;
			double Tz = paramsl[0] * paramsl[1] / (paramsl[2] * Ks[k].y + Ks[k].x);
			double Tx = (Ks[k].z - paramsl[2]) * paramsl[1] / (paramsl[0] * Ks[k].y + Ks[k].x);
			Cam_Pose(0, 0) = paramsl[1];//r11
			Cam_Pose(0, 1) = Tx;
			Cam_Pose(1, 0) = paramsl[3];//r31
			Cam_Pose(1, 1) = Tz;
			std::cout << "Cam_Pose No." << k + 1 << ":" << std::endl;
			std::cout << Cam_Pose << std::endl;
			std::cout << std::endl;
			Cam_Poses.push_back(Cam_Pose);
		

	}


}
//void SolveLine_Calibration(std::vector<double>& Distance, std::vector <cv::Point3d>& Ks, Eigen::Matrix2f& Cam_internal, std::vector< Eigen::Matrix2f>& Cam_Poses, std::vector <double>& High)
//{
//	double paramsl[4] = { 10000,1,4096,0 };
//	ceres::Problem probleml;
//	for (int i = 0; i < Ks.size()-1; i++)
//	{
//		ceres::CostFunction* cost_functionl = new ceres::AutoDiffCostFunction<LineFunctor, 4, 4>(new LineFunctor(Ks[0].x, Ks[0].y, Ks[0].z, Ks[i + 1].x, Ks[i + 1].y, Ks[i + 1].z, High[i], Distance[i]));
//
//		//三个参数分别为代价函数、核函数和待估参数
//		probleml.AddResidualBlock(cost_functionl, NULL, paramsl);
//	}
//	// 第二步，配置Solver  
//	ceres::Solver::Options optionsl;
//	//配置增量方程的解法
//	optionsl.max_num_iterations = 10000;
//	optionsl.linear_solver_type = ceres::DENSE_QR;
//	//是否输出到cout
//	optionsl.minimizer_progress_to_stdout = true;
//	//第三步，创建Summary对象用于输出迭代结果
//	ceres::Solver::Summary summaryl;
//	//第四步，执行求解
//	ceres::Solve(optionsl, &probleml, &summaryl);
//	std::cout << summaryl.BriefReport() << std::endl; // 添加此处
//	//内参
//	Cam_internal(0, 0) = paramsl[0];//Fu
//	Cam_internal(0, 1) = paramsl[2];//U0
//	Cam_internal(1, 0) = 0;
//	Cam_internal(1, 1) = 1;
//	std::cout << std::endl;
//	std::cout << "Cam_internal:" << std::endl;
//	std::cout << Cam_internal << std::endl;
//	std::cout << std::endl;
//	//外参
//	for (int i = 0; i < Ks.size(); i++)
//	{
//		Eigen::Matrix2f Cam_Pose;
//		double Tz = paramsl[0] * paramsl[1] / (paramsl[2] * Ks[i].y + Ks[i].x);
//		double Tx = (Ks[i].z - paramsl[2]) * paramsl[1] / (paramsl[0] * Ks[i].y + Ks[i].x);
//		Cam_Pose(0, 0) = paramsl[1];//r11
//		Cam_Pose(0, 1) = Tx;
//		Cam_Pose(1, 0) = paramsl[3];//r31
//		Cam_Pose(1, 1) = Tz;
//		std::cout << "Cam_Pose No." << i + 1 << ":" << std::endl;
//		std::cout << Cam_Pose << std::endl;
//		std::cout << std::endl;
//		Cam_Poses.push_back(Cam_Pose);
//	}
//}