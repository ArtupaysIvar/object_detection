/*
Enhanced ROI Obstacle Detector with Improved Detection Logic:
1. Soft flow analysis - logs status but doesn't block detection
2. Confidence scoring system instead of binary filters
3. Dynamic thresholds based on context
4. Relaxed static object suppression
5. Balanced detection sensitivity
6. Comprehensive logging for debugging
*/
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <deque>
#include <vector>
#include <numeric>

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
        detector_ = cv::ORB::create(600);
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false); //CEK

        // Initialize parameters (tunable)
        kp_ratio_threshold_ = 1.25;  // UBAH
        area_ratio_threshold_ = 1.35; // UBAH
        base_min_matches_ = 8;
        min_matches_floor_ = 4;
        match_ratio_threshold_ = 0.75;
        
        // Fixed center ROI parameters
        roi_margin_x_ = 0.25;
        roi_margin_y_ = 0.25;

        // Adaptive and emergency detection parameters
        adaptive_match_percentage_ = 0.45; // UBAH
        emergency_kp_threshold_ = 0.9;    // UBAH
        emergency_area_threshold_ = 0.85;   // UBAH
        trend_window_size_ = 3;
        strong_trend_threshold_ = 0.06;    // UBAH

        // IMPROVED: Soft flow analysis parameters
        flow_magnitude_min_threshold_ = 0.3;  // Lowered minimum
        flow_magnitude_max_threshold_ = 20.0; // Raised maximum
        flow_consistency_threshold_ = 0.15;   // Much more lenient
        
        // Detection consistency buffer parameters
        detection_buffer_size_ = 3;
        min_detections_for_trigger_ = 3; // UBAH

        // IMPROVED: Relaxed static object suppression
        static_area_growth_max_ = 1.15;      // Increased from 1.1
        static_kp_growth_max_ = 1.12;        // Increased from 1.08

        // NEW: Confidence scoring parameters
        min_confidence_threshold_ = 0.75;     // UBAH
        excellent_confidence_threshold_ = 0.85; // High confidence bypass

        RCLCPP_INFO(this->get_logger(), "Enhanced ROI obstacle detector with improved logic initialized."); 
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

    // Adaptive and emergency detection parameters
    double adaptive_match_percentage_;
    double emergency_kp_threshold_;
    double emergency_area_threshold_;
    int trend_window_size_;
    double strong_trend_threshold_;

    // Flow analysis parameters (now soft)
    double flow_magnitude_min_threshold_;
    double flow_magnitude_max_threshold_;
    double flow_consistency_threshold_;

    // Detection consistency buffer parameters
    int detection_buffer_size_;
    int min_detections_for_trigger_;

    // Static object suppression parameters (relaxed)
    double static_area_growth_max_;
    double static_kp_growth_max_;

    // NEW: Confidence scoring parameters
    double min_confidence_threshold_;
    double excellent_confidence_threshold_;

    // Trend tracking
    std::deque<double> kp_ratio_history_;
    std::deque<double> area_ratio_history_;

    // Detection consistency buffer
    std::deque<bool> detection_buffer_;
    std::deque<double> confidence_buffer_; // NEW: Track confidence history
    
    // Flow analysis storage
    std::vector<cv::Point2f> prev_flow_points_;

    // Adaptive min_matches calculation
    int calculateAdaptiveMinMatches(int available_keypoints) {
        int adaptive_matches = static_cast<int>(available_keypoints * adaptive_match_percentage_);
        adaptive_matches = std::max(adaptive_matches, min_matches_floor_);
        adaptive_matches = std::min(adaptive_matches, base_min_matches_);
        return adaptive_matches;
    }

    // Trend analysis functions
    void updateTrendHistory(double kp_ratio, double area_ratio) {
        kp_ratio_history_.push_back(kp_ratio);
        area_ratio_history_.push_back(area_ratio);
        
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
        
        return kp_growing || area_growing;
    }

    // IMPROVED: Soft flow analysis structure
    struct FlowAnalysis {
        double avg_magnitude;
        double flow_consistency;
        bool is_valid_motion;
        bool is_likely_shake;
        std::string status;
        double confidence_modifier; // NEW: How this affects confidence
    };

    // IMPROVED: Soft flow analysis - doesn't block, just informs
    FlowAnalysis analyzeOpticalFlow(const std::vector<cv::Point2f>& prev_pts, 
                                  const std::vector<cv::Point2f>& curr_pts) {
        FlowAnalysis analysis;
        analysis.avg_magnitude = 0.0;
        analysis.flow_consistency = 0.0;
        analysis.is_valid_motion = true;  // Default to valid
        analysis.is_likely_shake = false;
        analysis.status = "NO_FLOW";
        analysis.confidence_modifier = 0.0; // Neutral by default

        if (prev_pts.size() != curr_pts.size() || prev_pts.empty()) {
            return analysis;
        }

        // Calculate flow vectors and magnitudes
        std::vector<cv::Point2f> flow_vectors;
        std::vector<double> magnitudes;
        
        for (size_t i = 0; i < prev_pts.size(); ++i) {
            cv::Point2f flow = curr_pts[i] - prev_pts[i];
            flow_vectors.push_back(flow);
            double magnitude = cv::norm(flow);
            magnitudes.push_back(magnitude);
            analysis.avg_magnitude += magnitude;
        }
        
        analysis.avg_magnitude /= magnitudes.size();

        // Calculate flow consistency with dynamic threshold
        if (flow_vectors.size() > 1) {
            cv::Point2f mean_flow(0, 0);
            for (const auto& flow : flow_vectors) {
                mean_flow += flow;
            }
            mean_flow.x /= flow_vectors.size();
            mean_flow.y /= flow_vectors.size();

            double consistency_sum = 0.0;
            int valid_flows = 0;
            for (const auto& flow : flow_vectors) {
                if (cv::norm(flow) > 0.1 && cv::norm(mean_flow) > 0.1) {
                    double dot_product = flow.dot(mean_flow);
                    double cosine_similarity = dot_product / (cv::norm(flow) * cv::norm(mean_flow));
                    consistency_sum += std::max(0.0, cosine_similarity);
                    valid_flows++;
                }
            }
            if (valid_flows > 0) {
                analysis.flow_consistency = consistency_sum / valid_flows;
            }
        }

        // Dynamic flow consistency threshold based on number of points
        double dynamic_consistency_threshold = flow_consistency_threshold_;
        if (prev_pts.size() < 8) {
            dynamic_consistency_threshold *= 0.7; // More lenient for fewer points
        }

        // IMPROVED: Soft classification with confidence modifiers
        if (analysis.avg_magnitude < flow_magnitude_min_threshold_) {
            analysis.status = "SLOW_MOTION";
            analysis.confidence_modifier = -0.1; // Slight penalty
        } else if (analysis.avg_magnitude > flow_magnitude_max_threshold_) {
            analysis.is_likely_shake = true;
            analysis.status = "POSSIBLE_SHAKE";
            analysis.confidence_modifier = -0.2; // Moderate penalty
        } else if (analysis.flow_consistency < dynamic_consistency_threshold) {
            analysis.status = "INCONSISTENT_FLOW";
            analysis.confidence_modifier = -0.15; // Slight penalty
        } else {
            analysis.status = "GOOD_FLOW";
            analysis.confidence_modifier = 0.1; // Slight bonus
        }

        // Very extreme cases still affect validity
        if (analysis.avg_magnitude > flow_magnitude_max_threshold_ * 1.5) {
            analysis.is_valid_motion = false;
            analysis.confidence_modifier = -0.4; // Strong penalty
        }

        return analysis;
    }

    // IMPROVED: More nuanced static object detection
    bool isLikelyStaticObject(double avg_kp_ratio, double area_ratio, double flow_magnitude) {
        // If there's significant flow, it's probably not static
        if (flow_magnitude > flow_magnitude_min_threshold_ * 2) {
            return false;
        }
        
        // Relaxed thresholds for static detection
        return (avg_kp_ratio <= static_kp_growth_max_ && area_ratio <= static_area_growth_max_);
    }

    // NEW: Confidence scoring system
    struct DetectionConfidence {
        double score;
        std::string primary_reason;
        std::vector<std::string> contributing_factors;
        bool should_detect;
    };

    DetectionConfidence calculateDetectionConfidence(double avg_kp_ratio, double area_ratio, 
                                                   bool has_strong_trend, int match_count,
                                                   int adaptive_min_matches, const FlowAnalysis& flow,
                                                   bool is_static) {
        DetectionConfidence confidence;
        confidence.score = 0.0;
        confidence.should_detect = false;

        // Base confidence from growth ratios
        double kp_confidence = std::min(1.0, (avg_kp_ratio - 1.0) / (kp_ratio_threshold_ - 1.0));
        double area_confidence = std::min(1.0, (area_ratio - 1.0) / (area_ratio_threshold_ - 1.0));
        
        confidence.score = (kp_confidence + area_confidence) / 2.0;

        // Bonus for exceeding thresholds
        if (avg_kp_ratio >= kp_ratio_threshold_) {
            confidence.score += 0.2;
            confidence.contributing_factors.push_back("KP_THRESHOLD_MET");
        }
        if (area_ratio >= area_ratio_threshold_) {
            confidence.score += 0.2;
            confidence.contributing_factors.push_back("AREA_THRESHOLD_MET");
        }

        // Strong trend bonus
        if (has_strong_trend) {
            confidence.score += 0.25;
            confidence.contributing_factors.push_back("STRONG_TREND");
        }

        // Match quality bonus
        double match_quality = static_cast<double>(match_count) / adaptive_min_matches;
        if (match_quality > 1.5) {
            confidence.score += 0.1;
            confidence.contributing_factors.push_back("GOOD_MATCHES");
        }

        // Flow analysis modifier (soft)
        confidence.score += flow.confidence_modifier;
        if (flow.confidence_modifier != 0.0) {
            confidence.contributing_factors.push_back("FLOW_" + flow.status);
        }

        // Static object penalty (but not blocking)
        if (is_static) {
            confidence.score -= 0.2;
            confidence.contributing_factors.push_back("STATIC_PENALTY");
        }

        // Emergency detection bonus
        bool near_kp_threshold = avg_kp_ratio >= (kp_ratio_threshold_ * emergency_kp_threshold_);
        bool near_area_threshold = area_ratio >= (area_ratio_threshold_ * emergency_area_threshold_);
        if ((near_kp_threshold || near_area_threshold) && has_strong_trend) {
            confidence.score += 0.3;
            confidence.contributing_factors.push_back("EMERGENCY_CONDITIONS");
        }

        // Clamp confidence score
        confidence.score = std::max(0.0, std::min(1.0, confidence.score));

        // Determine primary reason and detection decision
        if (confidence.score >= excellent_confidence_threshold_) {
            confidence.primary_reason = "HIGH_CONFIDENCE";
            confidence.should_detect = true;
        } else if (confidence.score >= min_confidence_threshold_) {
            confidence.primary_reason = "MODERATE_CONFIDENCE";
            confidence.should_detect = true;
        } else {
            confidence.primary_reason = "LOW_CONFIDENCE";
            confidence.should_detect = false;
        }

        return confidence;
    }

    // IMPROVED: Detection consistency management with confidence
    void updateDetectionBuffer(bool detection, double confidence) {
        detection_buffer_.push_back(detection);
        confidence_buffer_.push_back(confidence);
        
        if (detection_buffer_.size() > detection_buffer_size_) {
            detection_buffer_.pop_front();
            confidence_buffer_.pop_front();
        }
    }

    bool shouldTriggerConsistentDetection(double& avg_confidence) {
        if (detection_buffer_.size() < detection_buffer_size_) {
            return false;
        }
        
        int detection_count = 0;
        double confidence_sum = 0.0;
        
        for (size_t i = 0; i < detection_buffer_.size(); ++i) {
            if (detection_buffer_[i]) detection_count++;
            confidence_sum += confidence_buffer_[i];
        }
        
        avg_confidence = confidence_sum / detection_buffer_.size();
        
        // Allow detection with fewer triggers if confidence is very high
        if (avg_confidence >= excellent_confidence_threshold_) {
            return detection_count >= 1; // Just need one high-confidence detection
        }
        
        return detection_count >= min_detections_for_trigger_;
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
            // curr_image = cv_bridge::toCvShare(msg, "bgr8")->image;
            curr_image = cv_bridge::toCvShare(msg, "mono8")->image; 
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // Grayscale conversion
        cv::Mat gray;
        // cv::cvtColor(curr_image, gray, cv::COLOR_BGR2GRAY);
        gray = curr_image.clone();
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
            publishDebugVisualization(curr_image, curr_keypoints, {}, {}, {}, 0, curr_roi, 
                                    msg->header, "INIT", {}, 0.0);
            return;
        }

        // Calculate adaptive min_matches based on available keypoints
        int min_available = std::min(static_cast<int>(prev_keypoints_.size()), static_cast<int>(curr_keypoints.size()));
        int adaptive_min_matches = calculateAdaptiveMinMatches(min_available);

        // More lenient feature check - allow processing with fewer features
        if (prev_descriptors_.empty() || curr_descriptors.empty() || 
            prev_keypoints_.size() < 3 || curr_keypoints.size() < 3) {
            
            RCLCPP_WARN(this->get_logger(), "Insufficient features: prev=%zu, curr=%zu", 
                       prev_keypoints_.size(), curr_keypoints.size());
            
            updateDetectionBuffer(false, 0.0);
            updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);
            publishDebugVisualization(curr_image, curr_keypoints, {}, {}, {}, 0, curr_roi, 
                                    msg->header, "LOW_FEATURES", {}, 0.0);
            return;
        }

        // Feature matching with kNN and ratio test
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(prev_descriptors_, curr_descriptors, knn_matches, 2);
        std::vector<cv::DMatch> good_matches = filterMatchesWithRatioTest(knn_matches);

        // More lenient match requirement
        int min_required_matches = std::max(3, adaptive_min_matches / 2);
        if (good_matches.size() < static_cast<size_t>(min_required_matches)) {
            RCLCPP_WARN(this->get_logger(), "Insufficient good matches: %zu (need %d)", 
                       good_matches.size(), min_required_matches);
            updateDetectionBuffer(false, 0.0);
            updatePreviousFrame(gray, curr_keypoints, curr_descriptors, curr_roi);
            publishDebugVisualization(curr_image, curr_keypoints, good_matches, {}, {}, 0, curr_roi, 
                                    msg->header, "LOW_MATCHES", {}, 0.0);
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

        // Analyze optical flow (soft analysis)
        FlowAnalysis flow_analysis = analyzeOpticalFlow(prev_pts, curr_pts);

        int obstacle_state = 0;
        std::vector<cv::Point2f> hull1, hull2;
        DetectionConfidence detection_confidence;
        detection_confidence.score = 0.0;
        detection_confidence.primary_reason = "INSUFFICIENT_DATA";

        if (prev_pts.size() >= 3) { // Lowered from 4
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

            // Update trend history
            updateTrendHistory(avg_kp_ratio, area_ratio);
            bool has_strong_trend = hasStrongGrowthTrend();

            // Check for static object (with flow consideration)
            bool is_static = isLikelyStaticObject(avg_kp_ratio, area_ratio, flow_analysis.avg_magnitude);

            // IMPROVED: Calculate confidence-based detection
            detection_confidence = calculateDetectionConfidence(avg_kp_ratio, area_ratio, has_strong_trend,
                                                              static_cast<int>(prev_pts.size()), adaptive_min_matches,
                                                              flow_analysis, is_static);

            RCLCPP_INFO(this->get_logger(), 
                "ROI: %dx%d, Expanding: %zu, AvgKP: %.3f, Area: %.3f, Flow: %s (%.2f), Confidence: %.3f (%s)", 
                curr_roi.width, curr_roi.height, prev_pts.size(), avg_kp_ratio, area_ratio, 
                flow_analysis.status.c_str(), flow_analysis.avg_magnitude, 
                detection_confidence.score, detection_confidence.primary_reason.c_str());
        }

        // Update detection buffer with confidence
        updateDetectionBuffer(detection_confidence.should_detect, detection_confidence.score);
        
        // Check for consistent detection
        double avg_confidence = 0.0;
        if (shouldTriggerConsistentDetection(avg_confidence)) {
            obstacle_state = 1;
            RCLCPP_WARN(this->get_logger(), "OBSTACLE DETECTED: %s (avg_conf: %.3f)", 
                       detection_confidence.primary_reason.c_str(), avg_confidence);
        }

        // Publish detection result
        std_msgs::msg::Int32 state_msg;
        state_msg.data = obstacle_state;
        detection_pub_->publish(state_msg);

        // Create and publish debug visualization
        publishDebugVisualization(curr_image, curr_keypoints, good_matches, hull1, hull2, 
                                obstacle_state, curr_roi, msg->header, 
                                detection_confidence.primary_reason, flow_analysis, 
                                detection_confidence.score);

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
                                 const std::string& reason,
                                 const FlowAnalysis& flow_analysis,
                                 double confidence)
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

        // Add status text with reason and confidence
        std::string status_text = obstacle_state ? ("OBSTACLE: " + reason) : reason;
        cv::Scalar text_color = obstacle_state ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(debug_img, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
        
        // Add confidence score with color coding
        cv::Scalar confidence_color = cv::Scalar(0, 255, 0); // Green for high confidence
        if (confidence < min_confidence_threshold_) {
            confidence_color = cv::Scalar(0, 255, 255); // Yellow for low confidence
        } else if (confidence >= excellent_confidence_threshold_) {
            confidence_color = cv::Scalar(0, 0, 255); // Red for excellent confidence
        }
        cv::putText(debug_img, "Confidence: " + std::to_string(confidence).substr(0, 4), 
                   cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 2);
        
        // Add detailed info
        cv::putText(debug_img, "ROI: " + std::to_string(roi.width) + "x" + std::to_string(roi.height), 
                   cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(debug_img, "Matches: " + std::to_string(matches.size()), 
                   cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(debug_img, "Keypoints: " + std::to_string(curr_keypoints.size()), 
                   cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        // Show flow analysis with appropriate color
        if (!flow_analysis.status.empty()) {
            cv::Scalar flow_color = cv::Scalar(0, 255, 0); // Green for good flow
            if (flow_analysis.is_likely_shake) {
                flow_color = cv::Scalar(0, 165, 255); // Orange for possible shake
            } else if (flow_analysis.status == "SLOW_MOTION") {
                flow_color = cv::Scalar(0, 255, 255); // Yellow for slow motion
            }
            
            cv::putText(debug_img, "Flow: " + flow_analysis.status + " (" + 
                       std::to_string(static_cast<int>(flow_analysis.avg_magnitude)) + "px)",
                       cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, flow_color, 1);
        }

        // Show trend indicator
        if (hasStrongGrowthTrend()) {
            cv::putText(debug_img, "STRONG GROWTH TREND", cv::Point(10, 160), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }

        // Show detection buffer status with confidence
        if (!detection_buffer_.empty()) {
            int detections = 0;
            double avg_conf = 0.0;
            for (size_t i = 0; i < detection_buffer_.size(); ++i) {
                if (detection_buffer_[i]) detections++;
                avg_conf += confidence_buffer_[i];
            }
            avg_conf /= detection_buffer_.size();
            
            cv::putText(debug_img, "Buffer: " + std::to_string(detections) + "/" + 
                       std::to_string(detection_buffer_.size()) + " (avg: " + 
                       std::to_string(avg_conf).substr(0, 4) + ")",
                       cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // Show confidence thresholds for reference
        cv::putText(debug_img, "Min/Exc Conf: " + std::to_string(min_confidence_threshold_).substr(0, 4) + 
                   "/" + std::to_string(excellent_confidence_threshold_).substr(0, 4),
                   cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(128, 128, 128), 1);

        // Publish debug image
        auto debug_msg = cv_bridge::CvImage(header, "bgr8", debug_img).toImageMsg();
        debug_image_pub_->publish(*debug_msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv); // Prepare ROS 2 system
    rclcpp::spin(std::make_shared<ROIObstacleDetectorNode>()); // Instantiate your custom node and Spin the node
    rclcpp::shutdown(); 
    return 0;
}