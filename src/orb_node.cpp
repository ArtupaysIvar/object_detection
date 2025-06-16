#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ORBTrackerNode : public rclcpp::Node
{
public:
  ORBTrackerNode() : Node("orb_tracker_node")
  {
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&ORBTrackerNode::image_callback, this, std::placeholders::_1));

    direction_pub_ = this->create_publisher<std_msgs::msg::String>("/avoid_direction", 10);

    orb_ = cv::ORB::create();
    bf_ = cv::BFMatcher(cv::NORM_HAMMING);

    RCLCPP_INFO(this->get_logger(), "ORB Tracker Node with direction logic started");
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

    if (!prev_keypoints_.empty() && !prev_descriptors_.empty()) {
      std::vector<cv::DMatch> matches;
      bf_.match(prev_descriptors_, descriptors, matches);

      float left_growth = 0.0, center_growth = 0.0, right_growth = 0.0;
      int left_count = 0, center_count = 0, right_count = 0;

      for (const auto& match : matches) {
        const auto& kp_prev = prev_keypoints_[match.queryIdx];
        const auto& kp_curr = keypoints[match.trainIdx];

        float growth = kp_curr.size / std::max(kp_prev.size, 1.0f);
        float x = kp_curr.pt.x;
        int width = gray.cols;

        if (x < width / 3) {
          left_growth += growth;
          left_count++;
        } else if (x < 2 * width / 3) {
          center_growth += growth;
          center_count++;
        } else {
          right_growth += growth;
          right_count++;
        }
      }

      float avg_left = left_count > 0 ? left_growth / left_count : 0.0f;
      float avg_center = center_count > 0 ? center_growth / center_count : 0.0f;
      float avg_right = right_count > 0 ? right_growth / right_count : 0.0f;

      std_msgs::msg::String direction_msg;

      if (avg_center > 1.2) {
        if (avg_left < avg_right && avg_left < 1.2) {
          direction_msg.data = "left";
        } else if (avg_right < 1.2) {
          direction_msg.data = "right";
        } else {
          direction_msg.data = "back";
        }
      } else {
        direction_msg.data = "forward";
      }

      direction_pub_->publish(direction_msg);
    }

    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr direction_pub_;
  cv::Ptr<cv::ORB> orb_;
  cv::BFMatcher bf_;

  std::vector<cv::KeyPoint> prev_keypoints_;
  cv::Mat prev_descriptors_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ORBTrackerNode>());
  rclcpp::shutdown();
  return 0;
}
