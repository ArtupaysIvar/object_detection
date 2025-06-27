/*
1. calculateFixedCenterROI
2. createROIMask
3. filterMatchesWithRatioTest
4. void imageCallback
5. void updatePreviousFrame
6. void publishDebugVisualization
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
            std::bind(&ROIObstacleDetectorNode::imageCallback, this, std::placeholders::_1));//panggil 1

        detection_pub_ = this->create_publisher<std_msgs::msg::Int32>("/obstacle_detected", 10);
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/debug/image_roi_orb", 10);

        // Use ORB for better performance and reliability
        detector_ = cv::ORB::create(800);  // Reduced since we're using ROI
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false); // Use kNN matching

        // Initialize parameters (tunable)
        kp_ratio_threshold_ = 1.15;
        /*
        Minimum average keypoint size ratio to trigger detection.
        If features (ORB keypoints) are larger in the current frame than in the previous frame, it indicates an object is getting closer.
        1.15 → 15% growth required.
        Higher value → only strong growth triggers obstacle
        Too low → false positives from minor changes
        */
        area_ratio_threshold_ = 1.3;
        /*
        Minimum convex hull area growth ratio.
        Convex hulls are built fro mmatched keypoints. If the hull grows in area, the object is moving toward the camera.
        1.3 → area must grow by 30%
        Filters noise from small scattered keypoints
        Often used with cv::contourArea(hull2) / cv::contourArea(hull1)
        */
        min_matches_ = 8; // Reduced for ROI
        /*
        Minimum number of valid expanding matches required to run detection logic.
        Avoids detecting based on just 1 or 2 keypoints, which is noisy and unreliable.
        Default: 8
        Ensures detection is based on a real cluster, not random motion
        If lower this too much, false positives increase.
        */
        match_ratio_threshold_ = 0.75; // For Lowe's ratio test
        
        // Fixed center ROI parameters
        roi_margin_x_ = 0.25; // 25% margin from sides (center 50% width)
        roi_margin_y_ = 0.25; // 25% margin from top/bottom (center 50% height)

        RCLCPP_INFO(this->get_logger(), "Fixed center ROI obstacle detector initialized.");
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

    // Tunable parameters
    double kp_ratio_threshold_;
    double area_ratio_threshold_;
    int min_matches_;
    double match_ratio_threshold_;
    
    // ROI parameters (fixed center approach)
    double roi_margin_x_;
    double roi_margin_y_;
    // Function to calculate fixed center ROI
    cv::Rect calculateFixedCenterROI(const cv::Mat& image)
    {
        int img_width = image.cols;
        int img_height = image.rows;
        
        // Fixed center ROI following Al-Kaff paper approach
        // Focus on center region where obstacles are most critical for UAV navigation
        int margin_x = static_cast<int>(img_width * roi_margin_x_);
        int margin_y = static_cast<int>(img_height * roi_margin_y_);
        
        int roi_x = margin_x;
        int roi_y = margin_y;
        int roi_width = img_width - 2 * margin_x;
        int roi_height = img_height - 2 * margin_y;
        
        return cv::Rect(roi_x, roi_y, roi_width, roi_height);
    }
    // Function to calculate fixed center ROI
    cv::Mat createROIMask(const cv::Mat& image, const cv::Rect& roi)
    {
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        mask(roi) = 255;
        return mask;
    }
    // Function to do Lowe's ratio test on kNN matches
    // kNN OEY
    std::vector<cv::DMatch> filterMatchesWithRatioTest(const std::vector<std::vector<cv::DMatch>>& knn_matches)
    {
        std::vector<cv::DMatch> good_matches;
        
        for (const auto& match_pair : knn_matches) {
            if (match_pair.size() == 2) {
                const cv::DMatch& m = match_pair[0];
                const cv::DMatch& n = match_pair[1];
                
                // Lowe's ratio test
                if (m.distance < match_ratio_threshold_ * n.distance) {
                    good_matches.push_back(m);
                }
            }
        }
        
        return good_matches;
    }
    // Function to update previous frame data
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

        // Apply Gaussian blur to reduce noise
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);

        // Calculate fixed center ROI
        cv::Rect curr_roi = calculateFixedCenterROI(gray); // Panggil 1
        cv::Mat roi_mask = createROIMask(gray, curr_roi); // Panggil 2

        // Feature detection using ORB within ROI
        std::vector<cv::KeyPoint> curr_keypoints;
        cv::Mat curr_descriptors;
        detector_->detectAndCompute(gray, roi_mask, curr_keypoints, curr_descriptors);

        // 
        if (!has_prev_frame_) {
            prev_image_ = gray.clone();
            /*
            Stores the current grayscale image as prev_image_ for use in the next frame.
            .clone() makes a deep copy so it won’t get overwritten.
            */
            prev_keypoints_ = curr_keypoints;
            // Stores the keypoints detected in this frame for next time
            prev_descriptors_ = curr_descriptors.clone();
            /*
            Stores the ORB feature descriptors for this frame.
            .clone() ensures we have a safe, separate copy
            */
            prev_roi_ = curr_roi;
            /*
            Stores the current Region of Interest (ROI) rectangle.
            Useful if you're visualizing or comparing ROI behavior across frames.
            */
            has_prev_frame_ = true;
            /*
            Marks that we have a valid "previous frame" now.
            On the next callback, it will skip this block and do real matching*/
            // Publish debug image for first frame
            publishDebugVisualization(curr_image, curr_keypoints, {}, {}, {}, 0, curr_roi, msg->header);// Panggil 6
            /*
            Shows current image, Draws keypoints, Draws ROI rectangle, Skips hulls or matches (empty {} used here), Marks it as "CLEAR" (obstacle_state = 0)
            */
            return;
        }

        // Skip if insufficient features
        if (prev_descriptors_.empty() || curr_descriptors.empty() || 
            prev_keypoints_.size() < min_matches_ || curr_keypoints.size() < min_matches_) {
            RCLCPP_WARN(this->get_logger(), "Insufficient features detected in ROI");
            updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);// Panggil 5
            publishDebugVisualization(curr_image, curr_keypoints, {}, {}, {}, 0, curr_roi, msg->header);// Panggil 6
            return;
        }

        // Feature matching with kNN and ratio test
        // kNN OEY
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(prev_descriptors_, curr_descriptors, knn_matches, 2);
        
        // Apply ratio test
        // kNN OEY
        std::vector<cv::DMatch> good_matches = filterMatchesWithRatioTest(knn_matches);// Panggil 3

        if (good_matches.size() < static_cast<size_t>(min_matches_)) {
            RCLCPP_WARN(this->get_logger(), "Insufficient good matches in ROI: %zu", good_matches.size());
            updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);// Panggil 5
            publishDebugVisualization(curr_image, curr_keypoints, good_matches, {}, {}, 0, curr_roi, msg->header); // Panggil 6
            return;
        }

        // Extract matched points and calculate size ratios
        std::vector<cv::Point2f> prev_pts, curr_pts;
        std::vector<double> size_ratios;
        
        for (const auto &match : good_matches) {
            const auto &prev_kp = prev_keypoints_[match.queryIdx];
            const auto &curr_kp = curr_keypoints[match.trainIdx];
            
            // Only consider points that are expanding (getting larger)
            double size_ratio = curr_kp.size / (prev_kp.size + 1e-6);
            if (size_ratio > 1.0) {
                prev_pts.push_back(prev_kp.pt);
                curr_pts.push_back(curr_kp.pt);
                size_ratios.push_back(size_ratio);
            }
        }

        int obstacle_state = 0;
        std::vector<cv::Point2f> hull1, hull2;

        if (prev_pts.size() >= 4) { // Need at least 4 points for meaningful hull
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

            RCLCPP_INFO(this->get_logger(), 
                "Center ROI: %dx%d, Expanding points: %zu, Avg KP ratio: %.2f, Area ratio: %.2f", 
                curr_roi.width, curr_roi.height, prev_pts.size(), avg_kp_ratio, area_ratio);
            
            // Obstacle detection logic (Al-Kaff paper approach)
            if (avg_kp_ratio >= kp_ratio_threshold_ && 
                area_ratio >= area_ratio_threshold_ && 
                prev_pts.size() >= static_cast<size_t>(min_matches_)) {
                obstacle_state = 1;
                RCLCPP_WARN(this->get_logger(), "OBSTACLE DETECTED in center ROI!");
            }
        }

        // Publish detection result
        std_msgs::msg::Int32 state_msg;
        state_msg.data = obstacle_state;
        detection_pub_->publish(state_msg);

        // Create and publish debug visualization
        publishDebugVisualization(curr_image, curr_keypoints, good_matches, hull1, hull2, 
                                obstacle_state, curr_roi, msg->header);

        // Update previous frame
        updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);// Panggil 5
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
                                 const std_msgs::msg::Header& header)
    {
        cv::Mat debug_img = curr_image.clone();
        
        // Draw ROI boundary
        cv::rectangle(debug_img, roi, cv::Scalar(255, 255, 0), 2); // Cyan ROI boundary
        
        // Draw keypoints (only those in ROI)
        cv::drawKeypoints(debug_img, curr_keypoints, debug_img, cv::Scalar(0, 255, 0), 
                         cv::DrawMatchesFlags::DEFAULT);

        // Draw convex hulls
        if (!hull1.empty()) {
            std::vector<std::vector<cv::Point>> hull_draw1 = {std::vector<cv::Point>(hull1.begin(), hull1.end())};
            cv::polylines(debug_img, hull_draw1, true, cv::Scalar(255, 0, 0), 2); // Blue for previous
        }

        if (!hull2.empty()) {
            std::vector<std::vector<cv::Point>> hull_draw2 = {std::vector<cv::Point>(hull2.begin(), hull2.end())};
            cv::polylines(debug_img, hull_draw2, true, cv::Scalar(0, 0, 255), 2); // Red for current
        }

        // Add status text
        std::string status_text = obstacle_state ? "OBSTACLE DETECTED" : "CLEAR";
        cv::Scalar text_color = obstacle_state ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(debug_img, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2);
        
        // Add ROI info
        cv::putText(debug_img, "Center ROI: " + std::to_string(roi.width) + "x" + std::to_string(roi.height), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(debug_img, "Matches: " + std::to_string(matches.size()), 
                   cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(debug_img, "Keypoints: " + std::to_string(curr_keypoints.size()), 
                   cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

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

