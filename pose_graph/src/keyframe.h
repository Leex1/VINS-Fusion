#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;


class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};

class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	//void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();



	double time_stamp; 
	int index;// 每addKeyFrame一次就加1，这个就是一个索引，所有通过降采样的关键帧都在里面排序,方便检查滑窗
	int local_index;// 在4dof优化问题中的索引，from最早的回环帧
	Eigen::Vector3d vio_T_w_i; // VIO节点的位姿【from 时间同步3buf】
	Eigen::Matrix3d vio_R_w_i; // 【from 时间同步3buf】
	Eigen::Vector3d T_w_i;// 在回环检测时是copy上面的vio_T_w_i，回环优化后是更新的位姿
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;// 和关键帧匹配的原图【from 时间同步3buf】
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d;// VIO世界坐标系下的地图点坐标【from 时间同步3buf】
	vector<cv::Point2f> point_2d_uv;// 归一化相机坐标系的坐标 【from 时间同步3buf】
	vector<cv::Point2f> point_2d_norm;// 像素坐标【from 时间同步3buf】
	vector<double> point_id;// 地图点idx【from 时间同步3buf】
	vector<cv::KeyPoint> keypoints;// 额外提的特征点的像素坐标系坐标
	vector<cv::KeyPoint> keypoints_norm;
	vector<cv::KeyPoint> window_keypoints;// 已有的特征点的像素坐标系坐标
	vector<BRIEF::bitset> brief_descriptors;// 额外提取的特征点的描述子
	vector<BRIEF::bitset> window_brief_descriptors;// 已有特征点的描述子
	bool has_fast_point;
	int sequence;

	bool has_loop;
	int loop_index;
	Eigen::Matrix<double, 8, 1 > loop_info;// 回环相对坐标转换T_loop_cur[xyz,Q,yaw]
};

