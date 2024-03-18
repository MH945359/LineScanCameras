#pragma once
#define _CRT_SECURE_NO_WARNINGS 
#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include "Eigen/Dense"
#include<iostream>
#include<string>
#include<fstream>
#include<algorithm>
#include<cstdlib>
#include<cmath>
struct KFunctor 
{

	double u_guan, xw_guan;

	//构造函数，用已知的x、y数据对其赋值
	 //CostFunctor(double xc, double u)
	KFunctor(double xw, double u)
	{
		xw_guan = xw;
		u_guan = u;
	}
	//重载括号运算符，两个参数分别是估计的参数和由该参数计算得到的残差
	template <typename T>
	bool operator()(const T* const params, T* residual)const
	{
		residual[0] = T(u_guan) - (params[0] * T(xw_guan) + params[1] * T(xw_guan) * T(u_guan) + params[2]);
		return true;
	}
};
struct LineFunctor
{
	double k1_guan, k2_guan, k3_guan, dL_yuan, k1i_guan, k2i_guan, k3i_guan, k1j_guan, k2j_guan, juli_guan;
	//构造函数，用已知的x、y数据对其赋值
	 //CostFunctor(double xc, double u)
	LineFunctor(double k1, double k2, double k3, double k11, double k21, double k31, double dL, double juli)
    //CostFunctorl(double k1, double k2, double k11, double k21, double k12, double k22, double jiaodu)
	{
		k1_guan = k1,
		k2_guan = k2;
		k3_guan = k3;
		k1i_guan = k11;
		k2i_guan = k21;
		k3i_guan = k31;
		juli_guan = juli;
		dL_yuan = dL;
	}
	//重载括号运算符，两个参数分别是估计的参数和由该参数计算得到的残差
	//注意这里的const，一个都不能省略，否则就会报错
	template <typename T>
	bool operator()(const T* const paramsl, T* residuals)const
	{
		T k1_guan_t = T(k1_guan);//垫高图像K1
		T k2_guan_t = T(k2_guan);//垫高图像K2
		T k3_guan_t = T(k3_guan);//垫高图像K3
		T k1i_guan_t = T(k1i_guan);//低图像
		T k2i_guan_t = T(k2i_guan);
		T k3i_guan_t = T(k3i_guan);
		T dL_yuan_t = T(dL_yuan);//升高多少
		T juli_t = T(juli_guan);//在空间上线移动多少
		/*paramsl[0] fu
		  paramsl[1] r11
		  paramsl[2] u0
		  paramsl[3] r31
		*/
		//residuals[0] = juli_t -ceres::abs((k3_guan_t - paramsl[2]) / (k2_guan_t * paramsl[2] + k1_guan_t) - (k3i_guan_t - paramsl[2]) / (k2i_guan_t * paramsl[2] + k1i_guan_t));
		////(Xwi-Xwj)=(k3i-U0)/(U0*k2i+k1i)-(k3j-U0)/(U0*k2j+k1j)
		//residuals[1] = 1.0 - (paramsl[1] * paramsl[1] + paramsl[3] * paramsl[3]);
		//residuals[2] = dL_yuan_t - ceres::abs(paramsl[1] * (paramsl[0] * paramsl[1] / (k2_guan_t * paramsl[2] + k1_guan_t) - paramsl[0] * paramsl[1] / (k2i_guan_t * paramsl[2] + k1i_guan_t)) - juli_t * paramsl[3]);
		////(FU*r11/(U0*k2i+k1i`)-FU*r11/(U0*k2j+k1j)-(Xwi-Xwj)*r31)*r11=dL
		//residuals[3] = paramsl[0] * paramsl[1] + paramsl[2] * paramsl[3] + k1_guan_t * paramsl[3] / k2_guan_t;
		////FU*r11+U0*r31+k1*r31/k2=0

		residuals[0] = dL_yuan_t - paramsl[1] * (paramsl[0] * paramsl[1] / (k2_guan_t * paramsl[2] + k1_guan_t) - paramsl[0] * paramsl[1] / (k2i_guan_t * paramsl[2] + k1i_guan_t) + juli_t * paramsl[3]);
		//(FU*r11/(U0*k2i+k1i`)-FU*r11/(U0*k2j+k1j)-(Xwi-Xwj)*r31)*r11=dL
		residuals[1] = (paramsl[0] * paramsl[1] + paramsl[2] * paramsl[3] + k1_guan_t * paramsl[3] / k2_guan_t);
		//FU*r11+U0*r31+k1*r31/k2=0
		residuals[2] = 1.0 - (paramsl[1] * paramsl[1] + paramsl[3] * paramsl[3]);
		residuals[3] = juli_t - ((k3_guan_t - paramsl[2]) / (k2_guan_t * paramsl[2] + k1_guan_t) - (k3i_guan_t - paramsl[2]) / (k2i_guan_t * paramsl[2] + k1i_guan_t));
		return true;
	}
};
void AreaSelect(int& number, cv::Mat& stats, cv::Mat& centroids, std::vector<cv::Point2d>& pix_point, int& colines);
void CalibrationConver(int linenum, std::vector<double>& VerPoint, std::vector<std::vector<cv::Point2d>>& HypPoints, std::vector<std::vector<double>>& Static_World_Points);
void SolveK(std::vector<std::vector<double>>& Static_World_Points, std::vector<std::vector<cv::Point2d>>& pix_points, std::vector <cv::Point3d>& Ks);
void SolveLine_Calibration(std::vector<cv::Point3d>& Dxyz, std::vector <cv::Point3d>& Ks, std::vector<Eigen::Matrix2d>& Cam_internals, std::vector<Eigen::Matrix2d>& Cam_Poses);
//void SolveLine_Calibration(std::vector<cv::Point3d>& Dxyz, std::vector <cv::Point3d>& Ks, std::vector<Eigen::Matrix2d>& Cam_internals, std::vector<Eigen::Matrix2d>& Cam_Poses, double high, double dx);
void SolveLine_Calibrationl(std::vector<double>& Dxyz, std::vector <cv::Point3d>& Ks, Eigen::Matrix2f& Cam_internal, std::vector<Eigen::Matrix2f>& Cam_Poses, double& high);
//void SolveLine_Calibration(std::vector<double>& Distance, std::vector <cv::Point3d>& Ks, Eigen::Matrix2f& Cam_internal, std::vector< Eigen::Matrix2f>& Cam_Poses, std::vector <double>& High);
