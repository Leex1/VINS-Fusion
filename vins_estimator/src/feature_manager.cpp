#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    // STEP1:遍历每个特征点,将其按照管理规则放入适当的位置
    for (auto &id_pts : image)
    {
        // 用特征点信息构造一个对象（因为每一帧都是一个新帧）
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

		// 在特征点清单(feature)中查找该特征点，是否能找到（曾经出现过）
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

		// TODO:从这里开始，理一下相机帧的数据结构啊！！
        if (it == feature.end())// 是一个新的特征点
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));// 创建一个新的特征点id。
            // 这里的frame_count就是该特征点在滑窗中的当前位置，作为这个特征点的起始位置(起始帧id)
            feature.back().feature_per_frame.push_back(f_per_fra);// 
        }
        else if (it->feature_id == feature_id)// 老特征点
        {
            it->feature_per_frame.push_back(f_per_fra);// 在对应的“组织”下增加一个帧属性
            last_track_num++;// 追踪到的上一帧的特征点数目
        }
    }
    // STEP2:遍历完该帧所有特征点后，判断是否是关键帧:前两帧都设置为关键帧，追踪过少也认为是KF
    if (frame_count < 2 || last_track_num < 20)// 注意此处和论文有出入
        return true;// 返回true程序结束

    // STEP3:如果既不是前两帧又没有追踪过少，则对于每个特征判断视差。平均视差超过阈值，也作为KF
    for (auto &it_per_id : feature)
    {
        // 计算的其实是frame_count-1帧，也就是前一帧是否为关键帧,计算前一帧需要用到再前一帧。即能不能被倒数第三帧看到，
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)// 如果这个特征是从倒数第三帧及之前开始的，且一直持续到倒数第二帧
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);// 计算倒数三和倒数二的某个特征点在归一化平面的距离
            parallax_num++;
        }
    }

    if (parallax_num == 0)// 上面的if没有进去，即倒三和倒二没有相同的特征点
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        // 看看平均视差是否超过一个阈值,是的话也作为一个关键帧
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

// 得到同时被f-1 和f-r看到的特征点 在各自帧下的坐标
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;// 前段光流去畸变之后的归一化相机坐标系下的坐标

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));// 返回相机坐标系下的坐标对
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);// 如果三角化求解失败，则erase这个特征点
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 利用观测到该特征点的所有位姿来三角化特征点
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        // Twi -> Twc,第一个观察到这个特征点的KF的位姿
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)// 能看到这个特征id的所有的KF
        {
            imu_j++;
            //得到该KF的相机坐标系位姿
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            // T_w_cj -> T_c0_cj，世界系转到枢纽帧系
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            // T_c0_cj -> T_cj_c0 相当于把c0当作世界坐标系的Twc ->Tcw
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // 构建超定方程中的两个方程（有点类似自己写的三角化？）
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        // 齐次化
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

		// 得到的深度值实际上就是 第一个观察到的这个特征点的相机坐标系下的深度值
        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;// 深度太小就设置成默认值
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

// 管理权的移交
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    // 遍历每一个特征点
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)// 如果该特征点不是被移除的帧看到，那么该地图点对应的起始帧id减去1
            it->start_frame--;// 除了被移除的帧，其他的全都往前移一帧
        else    // 则是原先被第0帧看到的
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  // 取出归一化相机坐标系坐标
            it->feature_per_frame.erase(it->feature_per_frame.begin());// 该点不再被原来的第一帧看到，因此从中移除
            if (it->feature_per_frame.size() < 2)// 如果这个地图点没有至少被两帧看到
            {
                feature.erase(it);// 那就没有存在价值
                continue;
            }
            else // 进行管辖权的转交
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;// 实际相机坐标系下的坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;// 转到世界坐标系下
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);// 转到新的最老帧的相机坐标系下
                double dep_j = pts_j(2);
                if (dep_j > 0)// 看看深度是否有效
                    it->estimated_depth = dep_j;// 有效的话就得到在现在最老帧下的深度值
                else
                    it->estimated_depth = INIT_DEPTH;// 无效就设置默认值
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

// 如果初始化还没有结束，不进行地图点新的深度的换算，因为此时还要进行视觉惯性的对齐
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 对margin倒数第二帧进行处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)    // 如果地图点被最后一帧开始看到，由于滑窗，他的起始帧前移
        {
            it->start_frame--;
        }
        else    // 该特征点不是从原最后一帧开始看到的
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;// 倒数第二帧在这个地图点对应KF vector的idx索引
            if (it->endFrame() < frame_count - 1)// 如果该地图点不能被倒数第二帧看到，那没用
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);// 能被倒数第二帧看到，erase掉这个索引(因为我们要边缘化这个次新帧)
            if (it->feature_per_frame.size() == 0)// 如果这个地图点没有别的观测了
                feature.erase(it);// 这个特征点废了
        }
    }
}

// 计算视差
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}