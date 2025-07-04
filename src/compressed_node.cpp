#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <deque>
#include <vector>
#include <numeric>

class compressedNode : public rclcpp::Node
{
public:
    compressedNode()
    : Node("compressedNode")
    {
        // image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    // 
            image_sub_ = image_transport::create_subscription(
                this,
                "/image_raw", 
                std::bind(&compressedNode::imageCallback, this, std::placeholders::_1),
                "compressed",
                rmw_qos_profile_sensor_data // Use SensorDataQoS for image data,
            );
        

        detection_pub_ = this->create_publisher<std_msgs::msg::Int32>("/obstacle_detected", 10);
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/debug_image_trial", 10);
        detector_ = cv::SIFT::create(600);
        matcher_ = cv::BFMatcher::create(cv::NORM_L2, false); //CEK
        roi_margin_x_ = 0.25; 
        roi_margin_y_ = 0.25; 

    }
    private:
   
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Subscriber image_sub_;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr detection_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    cv::Ptr<cv::SIFT> detector_;
    cv::Ptr<cv::BFMatcher> matcher_;
    // ROI parameters
    double roi_margin_x_;
    double roi_margin_y_;
    


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
        RCLCPP_INFO(this->get_logger(), "Image: %d x %d, ROI: x=%d y=%d w=%d h=%d",
            image.cols, image.rows,
            roi_x, roi_y, roi_width, roi_height);

        return cv::Rect(roi_x, roi_y, roi_width, roi_height);
    }

    cv::Mat createROIMask(const cv::Mat& image, const cv::Rect& roi)
    {
        cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
        mask(roi) = 255;
        return mask;
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) //const sensor_msgs::msg::Image::SharedPtr msg
    {
        RCLCPP_INFO(this->get_logger(), "Received image message with step %d and area %d x %d (%s)", msg->step, msg->width, msg->height, msg->encoding.c_str());
        cv::Mat curr_image;
        // cv::Mat curr_image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR); // Decode the compressed image data
        curr_image = cv_bridge::toCvShare(msg)->image; 
        RCLCPP_INFO(this->get_logger(), "Received image of size: %d x %d x %d, step: %u, payload size: %zu", 
            curr_image.cols, curr_image.rows, curr_image.channels(), msg->step, msg->data.size());
        cv::Mat gray_img;
        gray_img = curr_image.clone();
        // cv::GaussianBlur(gray_img, gray_img, cv::Size(5, 5), 1.0);
        cv::Rect curr_roi = calculateFixedCenterROI(gray_img);
        cv::Mat roi_mask = createROIMask(gray_img, curr_roi);
        // Feature detection using SIFT within ROI
        std::vector<cv::KeyPoint> curr_keypoints;
        cv::Mat curr_descriptors;
        detector_->detectAndCompute(gray_img, roi_mask, curr_keypoints, curr_descriptors);
        // Draw ROI boundary
        cv::rectangle(gray_img, curr_roi, cv::Scalar(255), 2);
        cv::drawKeypoints(gray_img, curr_keypoints, gray_img, cv::Scalar(255), 
                         cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        auto debug_msg = cv_bridge::CvImage(msg->header, "mono8", gray_img).toImageMsg();
        debug_image_pub_->publish(*debug_msg);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv); // Prepare ROS 2 system
    rclcpp::spin(std::make_shared<compressedNode>()); // Instantiate your custom node and Spin the node
    rclcpp::shutdown(); 
    return 0;
}