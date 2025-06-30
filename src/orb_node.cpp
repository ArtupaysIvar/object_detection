/*
Enhanced ROI Obstacle Detector with:
1. Adaptive min_matches_ based on keypoint availability
2. Last-second escape logic with trend analysis
3. Emergency detection when approaching critical thresholds
*/
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <deque>
#include <vector>

class ROIObstacleDetectorNode : public rclcpp::Node
{
public:
    ROIObstacleDetectorNode()
    : Node("roi_obstacle_detector_node")
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&ROIObstacleDetectorNode::imageCallback, this, std::placeholders::_1));

        detection_pub_ = this->create_publisher<std_msgs::msg::Int32>("/obstacle_detected", 10);
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/debug/image_roi_orb", 10);

        // Use ORB for better performance and reliability
        detector_ = cv::ORB::create(800);
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);

        // Initialize parameters (tunable)
        kp_ratio_threshold_ = 1.15;
        area_ratio_threshold_ = 1.3;
        base_min_matches_ = 8; // Base minimum matches
        min_matches_floor_ = 4; // Never go below this
        match_ratio_threshold_ = 0.75;
        
        // Fixed center ROI parameters
        roi_margin_x_ = 0.25;
        roi_margin_y_ = 0.25;

        // Adaptive and emergency detection parameters
        adaptive_match_percentage_ = 0.6; // Use 60% of available keypoints as min_matches
        emergency_kp_threshold_ = 0.9;    // 90% of kp_ratio_threshold_
        emergency_area_threshold_ = 0.85; // 85% of area_ratio_threshold_
        trend_window_size_ = 3;           // Track last 3 frames for trend
        strong_trend_threshold_ = 0.05;   // 5% consistent growth = strong trend

        RCLCPP_INFO(this->get_logger(), "Enhanced ROI obstacle detector with adaptive logic initialized.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr detection_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;

    cv::Mat prev_image_;
    std::vector<cv::KeyPoint> prev_keypoints_;
    cv::Mat prev_descriptors_;
    cv::Rect prev_roi_;
    bool has_prev_frame_ = false;

    cv::Ptr<cv::ORB> detector_;
    cv::Ptr<cv::BFMatcher> matcher_;

    // Base parameters
    double kp_ratio_threshold_;
    double area_ratio_threshold_;
    int base_min_matches_;
    int min_matches_floor_;
    double match_ratio_threshold_;
    
    // ROI parameters
    double roi_margin_x_;
    double roi_margin_y_;

    // NEW: Adaptive and emergency detection parameters
    double adaptive_match_percentage_;
    double emergency_kp_threshold_;
    double emergency_area_threshold_;
    int trend_window_size_;
    double strong_trend_threshold_;

    // NEW: Trend tracking
    std::deque<double> kp_ratio_history_;
    std::deque<double> area_ratio_history_;

    // NEW: Adaptive min_matches calculation
    int calculateAdaptiveMinMatches(int available_keypoints) {
        // Calculate based on percentage of available keypoints
        int adaptive_matches = static_cast<int>(available_keypoints * adaptive_match_percentage_);
        
        // Ensure it's between floor and base values
        adaptive_matches = std::max(adaptive_matches, min_matches_floor_);
        adaptive_matches = std::min(adaptive_matches, base_min_matches_);
        
        return adaptive_matches;
    }

    // NEW: Trend analysis functions
    void updateTrendHistory(double kp_ratio, double area_ratio) {
        // Add new values
        kp_ratio_history_.push_back(kp_ratio);
        area_ratio_history_.push_back(area_ratio);
        
        // Keep only recent history
        if (kp_ratio_history_.size() > trend_window_size_) {
            kp_ratio_history_.pop_front();
        }
        if (area_ratio_history_.size() > trend_window_size_) {
            area_ratio_history_.pop_front();
        }
    }

    bool hasStrongGrowthTrend() {
        if (kp_ratio_history_.size() < 2 || area_ratio_history_.size() < 2) {
            return false;
        }
        
        // Check if there's consistent growth in both metrics
        bool kp_growing = true;
        bool area_growing = true;
        
        for (size_t i = 1; i < kp_ratio_history_.size(); ++i) {
            if (kp_ratio_history_[i] - kp_ratio_history_[i-1] < strong_trend_threshold_) {
                kp_growing = false;
                break;
            }
        }
        
        for (size_t i = 1; i < area_ratio_history_.size(); ++i) {
            if (area_ratio_history_[i] - area_ratio_history_[i-1] < strong_trend_threshold_) {
                area_growing = false;
                break;
            }
        }
        
        return kp_growing || area_growing; // Either metric showing strong growth
    }

    // NEW: Emergency detection logic
    bool shouldTriggerEmergencyDetection(double avg_kp_ratio, double area_ratio, bool has_strong_trend) {
        // Check if we're approaching critical thresholds with strong growth trend
        bool near_kp_threshold = avg_kp_ratio >= (kp_ratio_threshold_ * emergency_kp_threshold_);
        bool near_area_threshold = area_ratio >= (area_ratio_threshold_ * emergency_area_threshold_);
        
        // Trigger if we're near any threshold AND have a strong growth trend
        return (near_kp_threshold || near_area_threshold) && has_strong_trend;
    }

    cv::Rect calculateFixedCenterROI(const cv::Mat& image)
    {
        int img_width = image.cols;
        int img_height = image.rows;
        
        int margin_x = static_cast<int>(img_width * roi_margin_x_);
        int margin_y = static_cast<int>(img_height * roi_margin_y_);
        
        int roi_x = margin_x;
        int roi_y = margin_y;
        int roi_width = img_width - 2 * margin_x;
        int roi_height = img_height - 2 * margin_y;
        
        return cv::Rect(roi_x, roi_y, roi_width, roi_height);
    }

    cv::Mat createROIMask(const cv::Mat& image, const cv::Rect& roi)
    {
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        mask(roi) = 255;
        return mask;
    }

    std::vector<cv::DMatch> filterMatchesWithRatioTest(const std::vector<std::vector<cv::DMatch>>& knn_matches)
    {
        std::vector<cv::DMatch> good_matches;
        
        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() == 2) {
                const cv::DMatch& m = match_pair[0];
                const cv::DMatch& n = match_pair[1];
                
                if (m.distance < match_ratio_threshold_ * n.distance) {
                    good_matches.push_back(m);
                }
            }
        }
        
        return good_matches;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert image
        cv::Mat curr_image;
        try {
            curr_image = cv_bridge::toCvShare(msg, "bgr8")->image;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // Grayscale conversion
        cv::Mat gray;
        cv::cvtColor(curr_image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);

        // Calculate fixed center ROI
        cv::Rect curr_roi = calculateFixedCenterROI(gray);
        cv::Mat roi_mask = createROIMask(gray, curr_roi);

        // Feature detection using ORB within ROI
        std::vector<cv::KeyPoint> curr_keypoints;
        cv::Mat curr_descriptors;
        detector_->detectAndCompute(gray, roi_mask, curr_keypoints, curr_descriptors);

        if (!has_prev_frame_) {
            prev_image_ = gray.clone();
            prev_keypoints_ = curr_keypoints;
            prev_descriptors_ = curr_descriptors.clone();
            prev_roi_ = curr_roi;
            has_prev_frame_ = true;
            publishDebugVisualization(curr_image, curr_keypoints, {}, {}, {}, 0, curr_roi, msg->header, "INIT");
            return;
        }

        // NEW: Calculate adaptive min_matches based on available keypoints
        int min_available = std::min(static_cast<int>(prev_keypoints_.size()), static_cast<int>(curr_keypoints.size()));
        int adaptive_min_matches = calculateAdaptiveMinMatches(min_available);

        // Skip if insufficient features (using adaptive threshold)
        if (prev_descriptors_.empty() || curr_descriptors.empty() || 
            prev_keypoints_.size() < adaptive_min_matches || curr_keypoints.size() < adaptive_min_matches) {
            
            RCLCPP_WARN(this->get_logger(), "Insufficient features: prev=%zu, curr=%zu, adaptive_min=%d", 
                       prev_keypoints_.size(), curr_keypoints.size(), adaptive_min_matches);
            
            updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);
            publishDebugVisualization(curr_image, curr_keypoints, {}, {}, {}, 0, curr_roi, msg->header, "LOW_FEATURES");
            return;
        }

        // Feature matching with kNN and ratio test
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(prev_descriptors_, curr_descriptors, knn_matches, 2);
        std::vector<cv::DMatch> good_matches = filterMatchesWithRatioTest(knn_matches);

        if (good_matches.size() < static_cast<size_t>(adaptive_min_matches)) {
            RCLCPP_WARN(this->get_logger(), "Insufficient good matches: %zu (need %d)", 
                       good_matches.size(), adaptive_min_matches);
            updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);
            publishDebugVisualization(curr_image, curr_keypoints, good_matches, {}, {}, 0, curr_roi, msg->header, "LOW_MATCHES");
            return;
        }

        // Extract matched points and calculate size ratios
        std::vector<cv::Point2f> prev_pts, curr_pts;
        std::vector<double> size_ratios;
        
        for (const auto &match : good_matches) {
            const auto &prev_kp = prev_keypoints_[match.queryIdx];
            const auto &curr_kp = curr_keypoints[match.trainIdx];
            
            double size_ratio = curr_kp.size / (prev_kp.size + 1e-6);
            if (size_ratio > 1.0) {
                prev_pts.push_back(prev_kp.pt);
                curr_pts.push_back(curr_kp.pt);
                size_ratios.push_back(size_ratio);
            }
        }

        int obstacle_state = 0;
        std::vector<cv::Point2f> hull1, hull2;
        std::string detection_reason = "CLEAR";

        if (prev_pts.size() >= 4) {
            // Calculate convex hulls
            cv::convexHull(prev_pts, hull1);
            cv::convexHull(curr_pts, hull2);

            double area1 = cv::contourArea(hull1);
            double area2 = cv::contourArea(hull2);

            // Calculate average keypoint size ratio
            double avg_kp_ratio = 0.0;
            for (double ratio : size_ratios) {
                avg_kp_ratio += ratio;
            }
            avg_kp_ratio /= size_ratios.size();

            double area_ratio = area2 / (area1 + 1e-6);

            // NEW: Update trend history
            updateTrendHistory(avg_kp_ratio, area_ratio);
            bool has_strong_trend = hasStrongGrowthTrend();

            RCLCPP_INFO(this->get_logger(), 
                "ROI: %dx%d, Expanding: %zu, AvgKP: %.3f, Area: %.3f, AdaptiveMin: %d, Trend: %s", 
                curr_roi.width, curr_roi.height, prev_pts.size(), avg_kp_ratio, area_ratio, 
                adaptive_min_matches, has_strong_trend ? "STRONG" : "weak");
            
            // NEW: Enhanced obstacle detection logic
            bool normal_detection = (avg_kp_ratio >= kp_ratio_threshold_ && 
                                   area_ratio >= area_ratio_threshold_ && 
                                   prev_pts.size() >= static_cast<size_t>(adaptive_min_matches));
            
            bool emergency_detection = shouldTriggerEmergencyDetection(avg_kp_ratio, area_ratio, has_strong_trend);
            
            if (normal_detection) {
                obstacle_state = 1;
                detection_reason = "NORMAL_DETECTION";
                RCLCPP_WARN(this->get_logger(), "OBSTACLE DETECTED (Normal): KP=%.3f, Area=%.3f", 
                           avg_kp_ratio, area_ratio);
            } else if (emergency_detection) {
                obstacle_state = 1;
                detection_reason = "EMERGENCY_DETECTION";
                RCLCPP_WARN(this->get_logger(), "OBSTACLE DETECTED (Emergency): KP=%.3f (%.1f%%), Area=%.3f (%.1f%%), Strong trend detected!", 
                           avg_kp_ratio, (avg_kp_ratio/kp_ratio_threshold_)*100,
                           area_ratio, (area_ratio/area_ratio_threshold_)*100);
            }
        }

        // Publish detection result
        std_msgs::msg::Int32 state_msg;
        state_msg.data = obstacle_state;
        detection_pub_->publish(state_msg);

        // Create and publish debug visualization
        publishDebugVisualization(curr_image, curr_keypoints, good_matches, hull1, hull2, 
                                obstacle_state, curr_roi, msg->header, detection_reason);

        // Update previous frame
        updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);
    }

    void updatePreviousFrame(const cv::Mat& gray, const std::vector<cv::KeyPoint>& keypoints, 
                           const cv::Mat& descriptors, const cv::Rect& roi)
    {
        prev_image_ = gray.clone();
        prev_keypoints_ = keypoints;
        prev_descriptors_ = descriptors.clone();
        prev_roi_ = roi;
    }

    void publishDebugVisualization(const cv::Mat& curr_image, 
                                 const std::vector<cv::KeyPoint>& curr_keypoints,
                                 const std::vector<cv::DMatch>& matches,
                                 const std::vector<cv::Point2f>& hull1,
                                 const std::vector<cv::Point2f>& hull2,
                                 int obstacle_state,
                                 const cv::Rect& roi,
                                 const std_msgs::msg::Header& header,
                                 const std::string& reason)
    {
        cv::Mat debug_img = curr_image.clone();
        
        // Draw ROI boundary
        cv::rectangle(debug_img, roi, cv::Scalar(255, 255, 0), 2);
        
        // Draw keypoints
        cv::drawKeypoints(debug_img, curr_keypoints, debug_img, cv::Scalar(0, 255, 0), 
                         cv::DrawMatchesFlags::DEFAULT);

        // Draw convex hulls
        if (!hull1.empty()) {
            std::vector<std::vector<cv::Point>> hull_draw1 = {std::vector<cv::Point>(hull1.begin(), hull1.end())};
            cv::polylines(debug_img, hull_draw1, true, cv::Scalar(255, 0, 0), 2);
        }

        if (!hull2.empty()) {
            std::vector<std::vector<cv::Point>> hull_draw2 = {std::vector<cv::Point>(hull2.begin(), hull2.end())};
            cv::polylines(debug_img, hull_draw2, true, cv::Scalar(0, 0, 255), 2);
        }

        // Add status text with reason
        std::string status_text = obstacle_state ? ("OBSTACLE: " + reason) : reason;
        cv::Scalar text_color = obstacle_state ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(debug_img, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2);
        
        // Add detailed info
        cv::putText(debug_img, "ROI: " + std::to_string(roi.width) + "x" + std::to_string(roi.height), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(debug_img, "Matches: " + std::to_string(matches.size()), 
                   cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(debug_img, "Keypoints: " + std::to_string(curr_keypoints.size()), 
                   cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        // NEW: Show trend indicator
        if (hasStrongGrowthTrend()) {
            cv::putText(debug_img, "STRONG GROWTH TREND", cv::Point(10, 120), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        }

        // Publish debug image
        auto debug_msg = cv_bridge::CvImage(header, "bgr8", debug_img).toImageMsg();
        debug_image_pub_->publish(*debug_msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ROIObstacleDetectorNode>());
    rclcpp::shutdown();
    return 0;
}
