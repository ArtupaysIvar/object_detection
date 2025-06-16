#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ORBDetectorNode : public rclcpp::Node
{
public:
  ORBDetectorNode() : Node("orb_detector_node")
  {
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&ORBDetectorNode::image_callback, this, std::placeholders::_1));

    obstacle_pub_ = this->create_publisher<std_msgs::msg::Bool>("/orb_obstacle_info", 10);

    orb_ = cv::ORB::create();
    RCLCPP_INFO(this->get_logger(), "ORB Detector Node started");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    // Draw keypoints for debug
    cv::Mat output;
    cv::drawKeypoints(gray, keypoints, output);
    cv::imshow("ORB Keypoints", output);
    cv::waitKey(1);

    // Publish obstacle detection if too many keypoints
    std_msgs::msg::Bool obstacle_msg;
    obstacle_msg.data = (keypoints.size() > 100);  // Simple threshold
    obstacle_pub_->publish(obstacle_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr obstacle_pub_;
  cv::Ptr<cv::ORB> orb_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ORBDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
